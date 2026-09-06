# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL prefill MoE for MiniMax-M3 MXFP4 weights on gfx950 (fp4 x fp4),
``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``.

At prefill sizes the aiter ``AITER_MXFP4_MXFP4`` path runs six kernels per
MoE layer: routing sort, per-token fp4 quant, stage-1 GEMM (bf16 out), a
separate fp4 quant of that intermediate, stage-2 GEMM writing the
routing-weighted bf16 partials ``[M, topk, H]``, and the top-k reduction.
This package keeps aiter's quant and reduction kernels and replaces the rest
with three FlyDSL kernels that produce the same bits:

* ``sort``: aiter's 3-stage ``moe_sorting`` contract, without the zero-fill
  of a ``[M, H]`` buffer the non-atomic path never reads (32768 tokens: 146 ->
  22 us);
* ``tile_map``: one CTA building gemm1's work list (expert-major, n-slab-major
  inside an expert, split by valid entries over the 8 XCDs);
* ``gemm1``: gate/up fp4 GEMM (4-wave 2x2, MFMA 16x16x128, AGPR accumulators)
  with the swiglu-OAI activation and the per-32-column e8m0 quant of the
  intermediate fused into the epilogue, exactly aiter's rounding;
* ``gemm2``: down fp4 GEMM, bf16 ``[M, topk, H]`` out with non-temporal stores
  and a rotated n-tile sweep so neighbouring CTAs of one expert share W2 in L2.

From 16384 tokens the sort uses 256-row blocks (gemm1 tiles 256 rows: half the
W13 bytes per FLOP; gemm2 keeps 128-row tiles and skips the all-padding ones).

Used only when every condition below holds; otherwise the layer keeps the
aiter path untouched:

* gfx950 and the ``AITER_MXFP4_MXFP4`` backend (``shuffle_weights`` +
  ``e8m0_shuffle`` weights, gate/up rows separated), the layout gemm1/gemm2
  read; ``AITER_MXFP4_BF16`` interleaves gate and up and is not supported;
* activation ``swigluoai_uninterleave`` with alpha 1.702, limit 7 and beta
  unset or 1 (compile-time constants of gemm1), no expert bias, no expert
  parallelism, ``apply_router_weight_on_input`` off;
* hidden size a multiple of 1024, per-partition intermediate size 768
  (gemm2's K pipeline is written for 3 K-steps of 256; MiniMax-M3 at TP4);
* at call time: ``3072 <= M <= 32768`` contiguous bf16 tokens and no unfused
  shared experts. Below 3072 tokens aiter's small-batch configuration is
  faster (512: 205 vs 289 us, 2048: 316 vs 326); the decode package covers
  ``M <= 256``.

MiniMax-M3 TP4 on MI355X, one MoE layer, HIP-graph replay of 4 different
inputs (us; aiter ``fused_moe`` in brackets): 4096 402 (473), 8192 614 (742),
16384 1059 (1328), 32768 1929 (2526) = 1.18 / 1.21 / 1.25 / 1.31x, the bf16
output bit-identical to aiter's.
"""

from __future__ import annotations

import functools

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

MIN_PREFILL_TOKENS = 3072
MAX_PREFILL_TOKENS = 32768
BM256_FROM_TOKENS = 16384  # sort / gemm1 block of 256 rows from here up
SORT_CTAS = 32
GEMM2_N_SPLIT = 2
GEMM1_SWIGLU_ALPHA = 1.702
GEMM1_SWIGLU_LIMIT = 7.0
_GEMM1_BLOCK_K = 256
_GEMM2_INTERMEDIATE = 768


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """Static shape gate: gemm1 unrolls the K loop by 4 steps of 256 after 4
    peeled ones; gemm2's flat pipeline is written for K = 768 (3 steps)."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % _GEMM1_BLOCK_K == 0
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size == _GEMM2_INTERMEDIATE
    )


def supports_batch(x: torch.Tensor) -> bool:
    """Runtime gate for one call (the layer keeps the aiter path otherwise)."""
    return (
        x.dim() == 2
        and MIN_PREFILL_TOKENS <= x.shape[0] <= MAX_PREFILL_TOKENS
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
    )


def block_m_for(n_tokens: int) -> int:
    return 256 if n_tokens >= BM256_FROM_TOKENS else 128


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction (the shim aiter ships in ``tensor_shim.py``)."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_sort(num_experts: int, topk: int, block_m: int):
    from .sort import compile_moe_sort

    return compile_moe_sort(
        E=num_experts, topk=topk, block_m=block_m, sort_ctas=SORT_CTAS
    )


@functools.cache
def _get_tile_map(intermediate_size: int, block_m: int):
    from .tile_map import compile_tile_map

    return compile_tile_map(I=intermediate_size, BM=block_m)


@functools.cache
def _get_gemm1(
    hidden_size: int, intermediate_size: int, num_experts: int, block_m: int
):
    from .gemm1 import compile_moe_gemm1

    return compile_moe_gemm1(
        H=hidden_size, I=intermediate_size, E=num_experts, BLOCK_M=block_m
    )


@functools.cache
def _get_gemm2(
    hidden_size: int, intermediate_size: int, num_experts: int, topk: int, block_m: int
):
    from .gemm2 import compile_moe_gemm2

    return compile_moe_gemm2(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        topk=topk,
        n_split=GEMM2_N_SPLIT,
        out_dtype="bf16",
        sort_block_m=block_m,
    )


def _u8_flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8).view(-1)


def a4w4_prefill_moe(
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
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One MoE layer for ``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``.

    ``w13`` ``[E, 2I, H/2]`` (gate rows then up rows, ``shuffle_weight(16,16)``),
    ``w2`` ``[E, H, I/2]`` and their ``e8m0_shuffle`` scales are the tensors the
    ``AITER_MXFP4_MXFP4`` backend stores on the layer; only their data pointers
    are used. ``topk_ids`` / ``topk_weights`` are ``[M, topk]`` with the fused
    shared expert included. Returns ``[M, hidden_size]`` bf16.
    """
    from aiter.ops.flydsl.moe_kernels import _run_moe_reduction
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

    from .gemm2 import gemm2_grid
    from .sort import SortBuffers
    from .tile_map import tile_map_grid

    n_tokens, hidden = x.shape
    assert MIN_PREFILL_TOKENS <= n_tokens <= MAX_PREFILL_TOKENS, n_tokens
    assert hidden == hidden_size and x.dtype == torch.bfloat16 and x.is_contiguous()
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = x.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)
    inter = intermediate_size

    # 1. routing sort (aiter moe_sorting contract, block size bm)
    bufs = SortBuffers.allocate(n_tokens, num_experts, topk, bm, SORT_CTAS, device)
    _get_sort(num_experts, topk, bm)(
        *bufs.launch_args(topk_ids, topk_weights, n_tokens)
    )
    # 2. aiter's per-token fp4 quant + activation scales in sorted, shuffled order
    a_q, a_s = fused_dynamic_mx_quant_moe_sort(
        x,
        bufs.sorted_ids,
        bufs.num_valid_ids,
        token_num=n_tokens,
        topk=topk,
        block_size=bm,
    )
    num_m_blocks = bufs.max_sorted // bm
    rows = num_m_blocks * bm
    # 3. gemm1 work list
    grid1 = tile_map_grid(num_m_blocks, inter)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=device)
    _run_compiled(
        _get_tile_map(inter, bm),
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        tile_map,
        num_m_blocks,
        grid1,
        stream,
    )
    # 4. gemm1: fp4 intermediate [rows, I/2] + e8m0 scales, sorted rows
    h_q = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
    h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
    _run_compiled(
        _get_gemm1(hidden_size, inter, num_experts, bm),
        _u8_flat(a_q),
        _u8_flat(w13),
        h_q.view(-1),
        _u8_flat(a_s),
        _u8_flat(w13_scale),
        h_s,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks,
        int(a_s.numel() * a_s.element_size()),
        tile_map,
        grid1,
        stream,
    )
    # 5. gemm2: routing-weighted bf16 partials [M, topk, H]
    partial = torch.empty(
        (n_tokens * topk, hidden_size), dtype=torch.bfloat16, device=device
    )
    partial_scale = torch.empty(
        (16,), dtype=torch.uint8, device=device
    )  # fp8 mode only
    num_m_blocks2 = rows // 128
    grid2 = gemm2_grid(num_m_blocks2, GEMM2_N_SPLIT)
    _run_compiled(
        _get_gemm2(hidden_size, inter, num_experts, topk, bm),
        h_q.view(-1),
        _u8_flat(w2),
        partial.view(-1),
        h_s,
        _u8_flat(w2_scale),
        partial_scale,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks2,
        grid2,
        stream,
    )
    # 6. aiter's top-k reduction (fp32 sum of the bf16 rows -> bf16)
    if out is None:
        out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    _run_moe_reduction(
        partial.view(n_tokens, topk, hidden_size), out, n_tokens, topk, hidden_size
    )
    return out


def _unsupported_reason(layer) -> str | None:
    """Static checks on a RoutedExperts layer; None when the fast path applies."""
    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode import (
        _backend_w13_layout,
    )
    from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode import (
        _unsupported_reason as decode_unsupported_reason,
    )

    reason = decode_unsupported_reason(layer)
    if reason is not None and not reason.startswith("shapes "):
        return reason
    qm = layer.quant_method
    if _backend_w13_layout(qm) != "standard":
        return "mxfp4 backend is not AITER_MXFP4_MXFP4 (gate/up-separated w13)"
    if (
        float(layer.swiglu_alpha) != GEMM1_SWIGLU_ALPHA
        or float(layer.swiglu_limit) != GEMM1_SWIGLU_LIMIT
    ):
        return (
            f"swiglu alpha/limit {layer.swiglu_alpha}/{layer.swiglu_limit} != 1.702/7"
        )
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition
    if not supports_shapes(hidden, inter):
        return f"shapes hidden={hidden} intermediate={inter} not tiled by the kernels"
    e = layer.w13_weight.shape[0]
    for name, t, shape in (
        ("w13_weight", layer.w13_weight, (e, 2 * inter, hidden // 2)),
        ("w2_weight", layer.w2_weight, (e, hidden, inter // 2)),
    ):
        if tuple(t.shape) != shape or t.element_size() != 1 or not t.is_contiguous():
            return (
                f"{name} {tuple(t.shape)} {t.dtype} is not the contiguous fp4 {shape}"
            )
    for name, t, numel in (
        ("w13_weight_scale", layer.w13_weight_scale, e * 2 * inter * hidden // 32),
        ("w2_weight_scale", layer.w2_weight_scale, e * hidden * inter // 32),
    ):
        if t.numel() != numel or t.element_size() != 1 or not t.is_contiguous():
            return f"{name} is not the contiguous e8m0 [{numel}] layout"
    return None


def install_prefill_fast_path(experts, prefix: str = "") -> bool:
    """Route ``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS`` calls of a
    MiniMax-M3 MoE layer to the FlyDSL a4w4 chain.

    ``experts`` is what ``FusedMoEFactory`` returned or the RoutedExperts layer
    itself. Wraps the layer's ``quant_method.apply`` (on top of the decode
    package's wrapper when that is installed); every call that fails the
    runtime gate goes to the wrapped implementation unchanged. Returns True
    when installed.
    """
    layer = getattr(experts, "routed_experts", experts)
    try:
        reason = _unsupported_reason(layer)
    except Exception as exc:  # a layer/config shape this gate does not know
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        logger.info_once(
            "M3 FlyDSL prefill MoE not used for %s: %s", prefix or "experts", reason
        )
        return False
    qm = layer.quant_method
    if getattr(qm, "_m3_prefill_fast_path", False):
        return True

    orig_apply = qm.apply
    hidden = layer.moe_config.hidden_dim
    inter = layer.moe_config.intermediate_size_per_partition

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
        return a4w4_prefill_moe(
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
        )

    qm.apply = apply
    qm._m3_prefill_fast_path = True
    logger.info_once(
        "M3 FlyDSL prefill MoE installed (%d <= M <= %d)",
        MIN_PREFILL_TOKENS,
        MAX_PREFILL_TOKENS,
    )
    logger.debug("M3 FlyDSL prefill MoE installed for %s", prefix or "experts")
    return True


__all__ = [
    "BM256_FROM_TOKENS",
    "MAX_PREFILL_TOKENS",
    "MIN_PREFILL_TOKENS",
    "a4w4_prefill_moe",
    "block_m_for",
    "install_prefill_fast_path",
    "supports_batch",
    "supports_shapes",
]
