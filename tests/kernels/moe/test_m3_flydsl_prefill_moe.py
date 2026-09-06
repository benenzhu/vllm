# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax-M3 FlyDSL prefill MoE (gfx950, MXFP4 weights and activations,
3072 <= M <= 32768).

Weights are quantized and shuffled like the AITER_MXFP4_MXFP4 backend does at
load time (shuffle_weights + e8m0_shuffle, gate/up rows separated), routing
mimics aiter's fused shared expert (4 distinct routed experts + expert 128,
weights renormalized x2). The output is compared with aiter's own a4w4 path
(``fused_moe`` per_1x32 / Swiglu / GateMode.SEPARATED, the production kernels
the chain replaces; both quantize the intermediate with the same e8m0 rule, so
the bf16 results are expected to be bit-identical) and, on a sample of tokens,
with a float reference on the dequantized experts.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]

HIDDEN, INTER, NUM_ROUTED, TOPK = 6144, 768, 128, 4
NUM_EXPERTS = NUM_ROUTED + 1
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0
CHECK_TOKENS = 64


def _quantize_like_aiter_mxfp4_backend(w13: torch.Tensor, w2: torch.Tensor):
    """per-1x32 MXFP4 + the shuffles oracle/mxfp4.py applies for
    AITER_MXFP4_MXFP4."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_f4_quant
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
    w13_k, w2_k = rocm_aiter_ops.shuffle_weights(w13_q.view(fp4), w2_q.view(fp4))
    w13_sk = e8m0_shuffle(w13_s.view(e * w13.shape[1], -1)).view(e, w13.shape[1], -1)
    w2_sk = e8m0_shuffle(w2_s.view(e * w2.shape[1], -1)).view(e, w2.shape[1], -1)
    return raw, (
        w13_k.contiguous(),
        w13_sk.contiguous(),
        w2_k.contiguous(),
        w2_sk.contiguous(),
    )


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


def _float_reference(x, raw, topk_ids, topk_weights, tokens):
    """Dequantized experts in fp32 on the sampled tokens, activations in
    fp32 (both implementations quantize x and the intermediate to fp4, so
    this is a loose reference: cos ~0.98)."""
    w13_q, w13_s, w2_q, w2_s = raw
    out = torch.zeros((len(tokens), HIDDEN), dtype=torch.float32, device=x.device)
    xf = x.float()
    for i, t in enumerate(tokens):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = xf[t] @ _dequant(w13_q[e], w13_s[e], HIDDEN).T
            g = h[:INTER].clamp(max=SWIGLU_LIMIT)
            u = h[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)
            w2e = _dequant(w2_q[e], w2_s[e], INTER)
            out[i] += float(topk_weights[t, j]) * (a @ w2e.T)
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
    return _quantize_like_aiter_mxfp4_backend(w13 * 0.02, w2 * 0.02)


def _run(shuffled, x, topk_ids, topk_weights):
    from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill import a4w4_prefill_moe

    w13_k, w13_sk, w2_k, w2_sk = shuffled
    return a4w4_prefill_moe(
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
    )


def _run_aiter(shuffled, x, topk_ids, topk_weights):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    w13_k, w13_sk, w2_k, w2_sk = shuffled
    return fused_moe(
        x,
        w13_k,
        w2_k,
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        w1_scale=w13_sk,
        w2_scale=w2_sk,
        swiglu_limit=SWIGLU_LIMIT,
        gate_mode=GateMode.SEPARATED.value,
    )


@pytest.mark.parametrize("m", [3072, 4096, 16384])
def test_prefill_moe_matches_aiter(m3_weights, m):
    """3072/4096: 128-row sort blocks; 16384: 256-row blocks (gemm1 BM256,
    gemm2 skipping the all-padding 128-row tiles)."""
    from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill import (
        block_m_for,
        supports_batch,
        supports_shapes,
    )

    assert supports_shapes(HIDDEN, INTER)
    assert block_m_for(m) == (256 if m >= 16384 else 128)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)
    assert supports_batch(x)

    out = _run(shuffled, x, topk_ids, topk_weights)
    out2 = _run(shuffled, x, topk_ids, topk_weights)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights)
    torch.cuda.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    # deterministic: the same inputs give the same bits (a race shows up here)
    assert torch.equal(out, out2)
    # aiter's a4w4 path applies the same quant rules; the bench sees 100%
    # bit-identical rows at these sizes, allow a whisker for other aiter builds
    same = (out == ref_aiter).float().mean().item()
    assert _cos(out, ref_aiter) > 0.9999
    assert same > 0.99, f"only {same:.4%} of the values match aiter"
    # both quantize x and the intermediate to fp4 (cos ~0.98 to the float
    # reference); ours must be as close to it as aiter's own kernels are
    tokens = torch.randperm(m, device=device)[:CHECK_TOKENS].tolist()
    ref = _float_reference(x, raw, topk_ids, topk_weights, tokens)
    ours = out[tokens].float()
    theirs = ref_aiter[tokens].float()
    assert _cos(ours, ref) > 0.97
    assert _cos(ours, ref) >= _cos(theirs, ref) - 1e-3
    err_ours = (ours - ref).abs().max().item()
    err_theirs = (theirs - ref).abs().max().item()
    assert err_ours <= err_theirs * 1.05 + 1e-3, (err_ours, err_theirs)


def test_prefill_moe_gate():
    from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill import (
        MAX_PREFILL_TOKENS,
        MIN_PREFILL_TOKENS,
        supports_batch,
        supports_shapes,
    )

    device = torch.device("cuda")
    assert supports_shapes(6144, 768)
    assert not supports_shapes(6144, 1536)  # gemm2 K pipeline is 3 x 256
    assert not supports_shapes(
        6144 + 256, 768
    )  # gemm1 unroll needs (K/256 - 4) % 4 == 0
    ok = torch.empty((MIN_PREFILL_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    assert supports_batch(ok)
    assert not supports_batch(ok[: MIN_PREFILL_TOKENS - 1])
    assert not supports_batch(ok.float())
    assert not supports_batch(ok.t())
    assert not supports_batch(
        torch.empty(
            (MAX_PREFILL_TOKENS + 1, HIDDEN), dtype=torch.bfloat16, device="meta"
        )
    )


def test_prefill_moe_defers_until_workspace_locked():
    """The runner sizes the modular kernel's shared workspace during the
    profile / warm-up / capture runs and locks it; the fast path must leave
    those runs to aiter (a batch that stays on aiter later would otherwise
    need a bigger workspace than the locked one)."""
    from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill import _workspace_locked
    from vllm.v1.worker import workspace as ws

    ws.reset_workspace_manager()
    assert _workspace_locked()  # no runner: nothing to size
    ws.init_workspace_manager(torch.device("cuda"))
    try:
        assert not _workspace_locked()
        ws.lock_workspace()
        assert _workspace_locked()
    finally:
        ws.reset_workspace_manager()
