# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 sort-free FlyDSL decode MoE (gfx950, MXFP4 weights, bf16 x).

Weights are quantized and shuffled exactly like the two aiter MXFP4 backends
do at load time (AITER_MXFP4_MXFP4: shuffle_weights + e8m0_shuffle, what the
Quark amd/MiniMax-M3-MXFP4 checkpoint gets; AITER_MXFP4_BF16:
shuffle_weight(is_guinterleave=True) + shuffle_scale), routing mimics aiter's
fused shared expert (4 distinct routed experts + expert 128, weights
renormalized x2), and the output is compared with a float reference on the
dequantized experts (the swiglu-OAI activation MiniMax-M3 is defined with:
alpha 1.702, clamp 7, up+1) and with aiter's CK a16w4 chain on the
gate/up-separated layout, which applies the same activation in its standalone
swiglu kernel.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]

# MiniMax-M3 at TP4: hidden 6144, intermediate 3072 / 4, 128 routed experts
# + 1 shared expert appended as id 128, top-4 routed + shared = 5 pairs.
HIDDEN, INTER, NUM_ROUTED, TOPK = 6144, 768, 128, 4
NUM_EXPERTS = NUM_ROUTED + 1
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0
LAYOUTS = ["standard", "guinterleave"]


def _quantize_like_aiter_backends(w13: torch.Tensor, w2: torch.Tensor):
    """per-1x32 MXFP4 + the shuffles oracle/mxfp4.py applies per aiter backend."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import (
        shuffle_scale,
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops

    e = w13.shape[0]
    fp4 = torch.float4_e2m1fn_x2
    w13_q, w13_s = per_1x32_f4_quant(w13, quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w13_q = w13_q.view(e, w13.shape[1], w13.shape[2] // 2)
    w2_q = w2_q.view(e, w2.shape[1], w2.shape[2] // 2)
    w13_s = w13_s.view(torch.uint8).view(e, w13.shape[1], w13.shape[2] // 32)
    w2_s = w2_s.view(torch.uint8).view(e, w2.shape[1], w2.shape[2] // 32)
    raw = (w13_q, w13_s, w2_q, w2_s)
    layouts = {}
    # AITER_MXFP4_MXFP4 (oracle/mxfp4.py): shuffle_weights + e8m0_shuffle
    w13_k, w2_k = rocm_aiter_ops.shuffle_weights(w13_q.view(fp4), w2_q.view(fp4))
    layouts["standard"] = (
        w13_k,
        e8m0_shuffle(w13_s.view(e * w13.shape[1], -1)).view(e, w13.shape[1], -1),
        w2_k,
        e8m0_shuffle(w2_s.view(e * w2.shape[1], -1)).view(e, w2.shape[1], -1),
    )
    # AITER_MXFP4_BF16: gate/up interleaved w13, shuffle_scale
    layouts["guinterleave"] = (
        shuffle_weight(w13_q.view(fp4), is_guinterleave=True, gate_up=True),
        shuffle_scale(w13_s.reshape(-1, w13_s.shape[-1]), e, True, True),
        shuffle_weight(w2_q.view(fp4), is_guinterleave=True, gate_up=False),
        shuffle_scale(w2_s.reshape(-1, w2_s.shape[-1]), e, True, False),
    )
    # gate/up-separated w13 for aiter's split CK chain (sort, fill, gemm1,
    # swiglu-OAI, gemm2); w2 and its scale are shared with the layouts above.
    sep = (
        shuffle_weight_a16w4(w13_q.view(fp4), 16, False),
        shuffle_scale_a16w4(w13_s.reshape(-1, w13_s.shape[-1]), e, False),
    )
    return raw, layouts, sep


def _routing(m: int, device):
    """4 distinct routed experts + the shared expert per token, like aiter's
    fused-shared-expert grouped top-k output (routed weights renormalized and
    scaled by routed_scaling_factor 2.0, shared weight 1)."""
    routed = torch.stack(
        [torch.randperm(NUM_ROUTED, device=device)[:TOPK] for _ in range(m)]
    )
    shared = torch.full((m, 1), NUM_ROUTED, device=device)
    topk_ids = torch.cat([routed, shared], dim=1).to(torch.int32)
    w = torch.rand((m, TOPK), device=device)
    w = w / w.sum(dim=1, keepdim=True) * 2.0
    topk_weights = torch.cat([w, torch.ones((m, 1), device=device)], dim=1)
    return topk_ids, topk_weights.to(torch.float32)


def _dequant(q: torch.Tensor, s: torch.Tensor, n_cols: int) -> torch.Tensor:
    from aiter.utility import fp4_utils

    v = fp4_utils.mxfp4_to_f32(q.view(torch.uint8)).view(q.shape[0], -1)
    sc = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).view(q.shape[0], -1)
    return (v.view(q.shape[0], -1, 32) * sc.unsqueeze(-1)).view(q.shape[0], n_cols)


def _float_reference(x, raw, topk_ids, topk_weights):
    w13_q, w13_s, w2_q, w2_s = raw
    out = torch.zeros((x.shape[0], HIDDEN), dtype=torch.float32, device=x.device)
    xf = x.float()
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = xf[t] @ _dequant(w13_q[e], w13_s[e], HIDDEN).T
            g = h[:INTER].clamp(max=SWIGLU_LIMIT)
            u = h[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)
            # the kernels round the stage-1 intermediate to bf16
            a = a.to(torch.bfloat16).float()
            w2e = _dequant(w2_q[e], w2_s[e], INTER)
            out[t] += float(topk_weights[t, j]) * (a @ w2e.T)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.fixture(scope="module")
def m3_weights():
    torch.manual_seed(0)
    device = torch.device("cuda")
    w13 = torch.randn(
        (NUM_EXPERTS, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device=device
    )
    w2 = torch.randn((NUM_EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16, device=device)
    return _quantize_like_aiter_backends(w13 * 0.02, w2 * 0.02)


def _run(layout_tensors, layout, x, topk_ids, topk_weights):
    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode import a16w4_decode_moe

    w13_k, w13_sk, w2_k, w2_sk = layout_tensors
    return a16w4_decode_moe(
        x,
        w13_k,
        w13_sk,
        w2_k,
        w2_sk,
        topk_weights,
        topk_ids,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_experts=NUM_EXPERTS,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_limit=SWIGLU_LIMIT,
        w13_layout=layout,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("m", [1, 2, 4, 8, 12, 16])
def test_decode_moe_matches_reference(m3_weights, layout, m):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode import (
        MAX_DECODE_TOKENS,
        supports_batch,
        supports_shapes,
    )

    assert supports_shapes(HIDDEN, INTER)
    raw, layouts, (w13_sep, w13_sep_s) = m3_weights
    torch.manual_seed(m)
    device = w13_sep.device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)
    assert supports_batch(x) and m <= MAX_DECODE_TOKENS

    out = _run(layouts[layout], layout, x, topk_ids, topk_weights)
    # aiter's split CK chain on the gate/up-separated layout (same experts,
    # same swiglu-OAI activation, independent implementation)
    _, _, w2_k, w2_sk = layouts["standard"]
    ref_aiter = fused_moe(
        x,
        w13_sep,
        w2_k,
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        w1_scale=w13_sep_s,
        w2_scale=w2_sk,
        swiglu_limit=SWIGLU_LIMIT,
        gate_mode=GateMode.SEPARATED.value,
    )
    torch.cuda.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    assert _cos(out, ref_aiter) > 0.999
    ref = _float_reference(x, raw, topk_ids, topk_weights)
    scale = ref.abs().max().item()
    assert _cos(out, ref) > 0.999
    assert (out.float() - ref).abs().max().item() < 0.02 * scale
    # the aiter chain itself agrees with the float reference just as well
    assert (ref_aiter.float() - ref).abs().max().item() < 0.02 * scale


def test_decode_moe_graph_replay(m3_weights):
    """HIP-graph capture with different routing per call (how vLLM runs it)."""
    _, layouts, _ = m3_weights
    tensors = layouts["standard"]
    torch.manual_seed(1)
    device = tensors[0].device
    m = 16
    inputs = [
        (
            torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device),
            *_routing(m, device),
        )
        for _ in range(4)
    ]

    eager = [_run(tensors, "standard", *inp).clone() for inp in inputs]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(tensors, "standard", *inputs[0])
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [_run(tensors, "standard", *inp) for inp in inputs]
    g.replay()
    g.replay()
    torch.cuda.synchronize()
    for got, want in zip(outs, eager):
        assert _cos(got, want) > 0.9999
