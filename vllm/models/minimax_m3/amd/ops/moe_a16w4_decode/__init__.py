# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL decode MoE for MiniMax-M3 MXFP4 weights on gfx950 (bf16 x), M <= 256.

The aiter CK path runs five kernels per MoE layer at decode batch sizes
(sort, zero-fill, gate/up GEMM, activation, down GEMM). At decode batch
sizes the whole layer is HBM-bound on the expert weights, so the launch and
sync gaps between those kernels are a large share of the time. This package
keeps two FlyDSL GEMM kernels (plus one small sort kernel above 16 tokens):

* ``gemm1``: gate/up GEMM (bf16 x MXFP4, MFMA 16x16x32) fused with the
  swiglu-OAI activation, bf16 out.
* ``gemm2``: down GEMM, split-K over 3 CTAs, routing-weighted bf16 atomic add
  into the (zeroed) output.

At ``M <= 16`` neither needs ``moe_sorting``: every m-block is one routing
pair ``(token, topk slot)`` and derives its expert id and its rows from
``topk_ids`` with a wave ballot (``n_tokens <= TILE_M`` rows fit one block; the
always-on shared expert owns all ``n_tokens`` rows, which pins ``TILE_M = 16``).
gemm1's first blocks zero the output so gemm2 can accumulate atomically.

At ``16 < M <= 256`` the same GEMMs run on expert-sorted rows: ``sort_decode``
(one launch: block 0 sorts the routing pairs by expert with LDS counters, the
other blocks zero the output; aiter's ``moe_sorting`` contract) feeds gemm1 and
gemm2 the per-block expert ids and token ids. Above 64 tokens gemm2 drops the
split-K (fewer atomics); aiter's own two sort kernels cost ~9.6 us here.

Only used when every condition below holds; otherwise the layer keeps the
aiter path untouched:

* gfx950 and an aiter MXFP4 CK MoE backend whose weight layout the kernels
  read: ``AITER_MXFP4_MXFP4`` (Quark ``w_mxfp4_a_mxfp4`` checkpoints such as
  amd/MiniMax-M3-MXFP4; ``shuffle_weights`` + ``e8m0_shuffle``, and aiter
  itself runs bf16 activations below 256 tokens) or ``AITER_MXFP4_BF16``
  (``shuffle_weight(is_guinterleave=True)`` + ``shuffle_scale``);
* activation ``swigluoai_uninterleave`` with ``swiglu_beta`` unset or 1, no
  expert bias, no expert parallelism, ``apply_router_weight_on_input`` off;
* hidden size a multiple of 256, per-partition intermediate size a multiple
  of 768 (MiniMax-M3 at TP4: 6144 / 768);
* at call time: ``M <= 256`` contiguous bf16 tokens and no unfused shared
  experts (the shared expert must be folded into ``topk_ids``).

MiniMax-M3 TP4 on MI355X, HIP-graph replay, 100 different inputs per graph
(us per layer, aiter production path in brackets: CK a16w4 below 256 tokens,
FlyDSL a4w4 with fp4 activations at 256): M=4 26.5 (42.4), M=8 44.3 (60.2),
M=12 56.8 (74.8), M=16 69.1 (89.4), M=32 104.8 (121.5), M=64 137.8 (153.3),
M=128 161.7 (181.0), M=256 181.6 (200.3). bf16 activations throughout (cos
0.99999 to the float reference; aiter's fp4-activation path at 256 is 0.97).
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# One m-block per routing pair holds every token of the batch: M <= TILE_M
# takes the sort-free path, larger batches the sorted one.
TILE_M = 16
MAX_PAIRS_TOKENS = TILE_M
MAX_DECODE_TOKENS = 256
# gemm2 split-K: worth ~1.5 us of latency hiding at small M, costs ~5 us of
# extra atomics at M=256 (same-GPU sweeps at 32/64 vs 128/256).
GEMM2_KSPLIT_SMALL_M = 64

# Best configuration from the FlyDSL sweep (see the module docstring).
GEMM1_CFG = dict(
    tile_m=TILE_M,
    tile_n=32,
    tile_k=128,
    k_wave=4,
    b_nt=2,
    xcd_swizzle=0,
    a_direct=True,
    prefetch=3,
    scale_share=True,
    act="swigluoai",
)
GEMM2_CFG = dict(
    tile_m=TILE_M,
    tile_n=256,
    tile_k=256,
    b_nt=2,
    xcd_swizzle=0,
    ksplit=3,
    pad_mask=True,
    hoist=True,
)
_GEMM2_K_UNIT = GEMM2_CFG["tile_k"] * GEMM2_CFG["ksplit"]
_GEMM2_N_UNIT = GEMM2_CFG["tile_n"]

# mxfp4 backend -> gemm1 weight layout (w2 and the scales are laid out the same
# way by both backends: shuffle_weight(16, 16) and e8m0_shuffle).
_BACKEND_W13_LAYOUT = {
    "AITER_MXFP4_MXFP4": "standard",
    "AITER_MXFP4_BF16": "guinterleave",
}

_workspaces: dict[tuple[int, int, int], torch.Tensor] = {}


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """Static shape gate: gemm2 tiles N by 256 and K by 256 x split-K 3."""
    return hidden_size % _GEMM2_N_UNIT == 0 and intermediate_size % _GEMM2_K_UNIT == 0


def supports_batch(x: torch.Tensor) -> bool:
    """Runtime gate for one call (the layer keeps the aiter path otherwise)."""
    return (
        x.dim() == 2
        and x.shape[0] <= MAX_DECODE_TOKENS
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def _intermediate_workspace(
    device: torch.device, topk: int, intermediate_size: int, num_experts: int
) -> torch.Tensor:
    """gemm1 output, bf16 ``[rows, I]``: by pair (``pair * TILE_M + row``) on the
    sort-free path, by sorted row on the sorted one; sized for the larger of
    the two at ``MAX_DECODE_TOKENS`` and allocated once per (device, topk, I,
    E) so HIP-graph capture records no allocation.
    """
    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode.sort_decode import (
        max_sorted_rows,
    )

    key = (
        device.index if device.index is not None else -1,
        topk,
        intermediate_size,
        num_experts,
    )
    ws = _workspaces.get(key)
    if ws is None:
        rows = max(
            MAX_PAIRS_TOKENS * topk * TILE_M,
            max_sorted_rows(MAX_DECODE_TOKENS, num_experts, topk, TILE_M),
        )
        ws = torch.empty((rows, intermediate_size), dtype=torch.bfloat16, device=device)
        _workspaces[key] = ws
    return ws


def a16w4_decode_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    swiglu_alpha: float,
    swiglu_limit: float,
    w13_layout: str = "standard",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One MoE layer for ``M <= 256`` tokens on the FlyDSL kernels.

    ``w13``/``w2`` and their e8m0 scales are the tensors the aiter MXFP4 backend
    stores on the layer; only their data pointers are used. ``w13_layout`` is
    ``"standard"`` for ``shuffle_weights`` + ``e8m0_shuffle`` (AITER_MXFP4_MXFP4)
    or ``"guinterleave"`` for ``shuffle_weight(is_guinterleave=True)`` +
    ``shuffle_scale`` (AITER_MXFP4_BF16). ``topk_ids`` and ``topk_weights`` are
    ``[M, topk]`` and already include the fused shared expert. Returns
    ``[M, hidden_size]`` bf16.
    """
    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode.host import (
        a16w4_gemm1,
        a16w4_gemm2,
    )

    n_tokens = x.shape[0]
    assert n_tokens <= MAX_DECODE_TOKENS, n_tokens
    assert x.shape[1] == hidden_size and x.dtype == torch.bfloat16 and x.is_contiguous()
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    inter = _intermediate_workspace(x.device, topk, intermediate_size, num_experts)
    if out is None:
        out = torch.empty(
            (n_tokens, hidden_size), dtype=torch.bfloat16, device=x.device
        )
    if n_tokens > MAX_PAIRS_TOKENS:
        return _sorted_decode_moe(
            x,
            w13,
            w13_scale,
            w2,
            w2_scale,
            topk_weights,
            topk_ids,
            inter,
            out,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            swiglu_alpha=swiglu_alpha,
            swiglu_limit=swiglu_limit,
            w13_layout=w13_layout,
        )
    # gemm1 zeroes `out` (its pair-0 blocks) before gemm2's atomics.
    a16w4_gemm1(
        x_bf16=x,
        w1_u8=w13,
        w1_scale_u8=w13_scale,
        inter_sorted_bf16=inter,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        topk=topk,
        pairs=True,
        topk_ids=topk_ids,
        zero_out=out,
        alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        w_layout=w13_layout,
        **GEMM1_CFG,
    )
    a16w4_gemm2(
        inter_sorted_bf16=inter,
        w2_u8=w2,
        w2_scale_u8=w2_scale,
        out_bf16=out,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        pairs=True,
        topk=topk,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        **GEMM2_CFG,
    )
    return out


def _sorted_decode_moe(
    x,
    w13,
    w13_scale,
    w2,
    w2_scale,
    topk_weights,
    topk_ids,
    inter,
    out,
    *,
    hidden_size,
    intermediate_size,
    num_experts,
    swiglu_alpha,
    swiglu_limit,
    w13_layout,
):
    """``16 < M <= 256``: sort_decode (sort + zero `out`) -> gemm1 -> gemm2 on
    expert-sorted rows (aiter ``moe_sorting`` contract, TILE_M-row blocks)."""
    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode.host import (
        a16w4_gemm1,
        a16w4_gemm2,
    )
    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode.sort_decode import (
        moe_sort_decode,
    )

    n_tokens, topk = topk_ids.shape
    sorted_ids, sorted_w, sorted_eids, num_valid, _ = moe_sort_decode(
        topk_ids, topk_weights, num_experts, hidden_size, TILE_M, out=out
    )
    a16w4_gemm1(
        x_bf16=x,
        w1_u8=w13,
        w1_scale_u8=w13_scale,
        sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid,
        sorted_token_ids=sorted_ids,
        inter_sorted_bf16=inter,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        topk=topk,
        alpha=swiglu_alpha,
        swiglu_limit=swiglu_limit,
        w_layout=w13_layout,
        **GEMM1_CFG,
    )
    cfg2 = dict(GEMM2_CFG)
    if n_tokens > GEMM2_KSPLIT_SMALL_M:
        cfg2["ksplit"] = 1
    a16w4_gemm2(
        inter_sorted_bf16=inter,
        w2_u8=w2,
        w2_scale_u8=w2_scale,
        sorted_expert_ids=sorted_eids,
        num_valid_ids=num_valid,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_w,
        out_bf16=out,
        n_tokens=n_tokens,
        NE=num_experts,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        **cfg2,
    )
    return out


def _backend_w13_layout(quant_method) -> str | None:
    """gemm1 weight layout for the layer's mxfp4 backend, None if unsupported."""
    backend = getattr(quant_method, "mxfp4_backend", None)
    return _BACKEND_W13_LAYOUT.get(getattr(backend, "value", None))


def _unsupported_reason(layer) -> str | None:
    """Static checks on a RoutedExperts layer; None when the fast path applies."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.platforms.rocm import on_gfx950

    if not on_gfx950():
        return "requires gfx950"
    qm = getattr(layer, "quant_method", None)
    if _backend_w13_layout(qm) is None:
        backend = getattr(qm, "mxfp4_backend", None)
        return (
            f"quant method {type(qm).__name__} (mxfp4 backend {backend}) is not "
            "AITER_MXFP4_MXFP4 or AITER_MXFP4_BF16"
        )
    if layer.activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
        return f"activation is {layer.activation}"
    if layer.swiglu_alpha is None or layer.swiglu_limit is None:
        return "swiglu_alpha / swiglu_limit not set"
    if layer.swiglu_beta not in (None, 1.0):
        return f"swiglu_beta {layer.swiglu_beta} != 1"
    if layer.apply_router_weight_on_input:
        return "apply_router_weight_on_input"
    if layer.expert_map is not None or layer.moe_config.use_ep:
        return "expert parallelism"
    if layer.moe_config.has_bias:
        return "expert bias"
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition
    if not supports_shapes(hidden, inter):
        return f"shapes hidden={hidden} intermediate={inter} not tiled by the kernels"
    if layer.moe_config.hidden_dim_unpadded not in (None, hidden):
        return "padded hidden size"
    if layer.moe_config.intermediate_size_per_partition_unpadded not in (None, inter):
        return "padded intermediate size"
    return None


def install_decode_fast_path(experts, prefix: str = "") -> bool:
    """Route ``M <= 256`` calls of a MiniMax-M3 MoE layer to the FlyDSL kernels.

    ``experts`` is what ``FusedMoEFactory`` returned (a runner holding
    ``routed_experts``) or the RoutedExperts layer itself. Wraps the layer's
    ``quant_method.apply``; every call that fails the runtime gate goes to the
    original aiter implementation unchanged. Returns True when installed.
    """
    layer = getattr(experts, "routed_experts", experts)
    try:
        reason = _unsupported_reason(layer)
    except Exception as exc:  # a layer/config shape this gate does not know
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        logger.info_once(
            "M3 FlyDSL decode MoE not used for %s: %s", prefix or "experts", reason
        )
        return False
    qm = layer.quant_method
    if getattr(qm, "_m3_decode_fast_path", False):
        return True

    orig_apply = qm.apply
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition
    alpha = float(layer.swiglu_alpha)
    limit = float(layer.swiglu_limit)
    w13_layout = _backend_w13_layout(qm)

    def apply(
        layer,
        x,
        topk_weights,
        topk_ids,
        shared_experts=None,
        shared_experts_input=None,
        **kwargs,
    ):
        if shared_experts is not None or kwargs or not supports_batch(x):
            return orig_apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
                **kwargs,
            )
        return a16w4_decode_moe(
            x,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
            topk_weights,
            topk_ids,
            hidden_size=hidden,
            intermediate_size=inter,
            num_experts=layer.w13_weight.shape[0],
            swiglu_alpha=alpha,
            swiglu_limit=limit,
            w13_layout=w13_layout,
        )

    qm.apply = apply
    qm._m3_decode_fast_path = True
    logger.info_once(
        "M3 FlyDSL decode MoE installed (M <= %d, w13 layout %s)",
        MAX_DECODE_TOKENS,
        w13_layout,
    )
    logger.debug("M3 FlyDSL decode MoE installed for %s", prefix or "experts")
    return True


__all__ = [
    "MAX_DECODE_TOKENS",
    "MAX_PAIRS_TOKENS",
    "TILE_M",
    "a16w4_decode_moe",
    "install_decode_fast_path",
    "supports_batch",
    "supports_shapes",
]
