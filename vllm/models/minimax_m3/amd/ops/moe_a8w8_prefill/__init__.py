# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FlyDSL prefill MoE for MiniMax-M3 MXFP8 weights on gfx950 (fp8 x fp8),
``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``.

The MXFP8 layer (``ModelOptMxFp8FusedMoE``, backend AITER_MXFP8) runs aiter's
a8w8 chain: routing sort, fused per-token fp8 quant, stage-1 GEMM writing the
fp8 intermediate + e8m0 scales, stage-2 GEMM (bf16 ``[M, topk, H]`` partials
or atomics) and the top-k reduction. This package keeps aiter's quant and
reduction kernels and the ``moe_a4w4_prefill`` sort / tile map, and replaces
the two GEMMs with the fp8 ports of the a4w4 prefill kernels:

* ``gemm1``: gate/up fp8 GEMM (4-wave 2x2, ``v_mfma_scale_f32_16x16x128_f8f6f4``
  with fp8 operands, AGPR accumulators, 128-K steps) with swiglu-OAI and the
  per-32-column MXFP8 quant of the intermediate fused into the epilogue; reads
  the gate/up-interleaved W13 the AITER_MXFP8 backend stores;
* ``gemm2``: down fp8 GEMM, bf16 ``[M, topk, H]`` out (to be written).

Weights: ``w13`` ``[E, 2I, H]`` fp8 e4m3 (``shuffle_weight(is_guinterleave=True,
gate_up=True)``), ``w13_scale`` (``shuffle_scale(..., True, True)``), ``w2``
``[E, H, I]`` (``shuffle_weight``), ``w2_scale`` (``shuffle_scale``): the tensors
``convert_to_fp8_moe_kernel_format`` leaves on the layer. Only their data
pointers are used.
"""

from __future__ import annotations

import functools

import torch

from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill import (
    MAX_PREFILL_TOKENS,
    MIN_PREFILL_TOKENS,
    _get_sort,
    _get_tile_map,
    _run_compiled,
    block_m_for,
)

GEMM1_SWIGLU_ALPHA = 1.702
GEMM1_SWIGLU_LIMIT = 7.0
GEMM2_N_SPLIT = 4  # 32768 tokens: 1285 -> 1088 us with the rotated sweep; 4096 unchanged
_GEMM1_BLOCK_K = 128
_GEMM2_INTERMEDIATE = 768


def _u8_flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8).view(-1)


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """gemm1 unrolls the K loop by 4 steps of 128 after 4 peeled ones; gemm2's
    pipeline is written for K = 768."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % 256 == 0
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size == _GEMM2_INTERMEDIATE
    )


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
        sort_block_m=block_m,
    )


def a8w8_prefill_moe(
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
    """One MoE layer for ``MIN_PREFILL_TOKENS <= M <= MAX_PREFILL_TOKENS``:
    stage 1 (sort, aiter fp8 quant, tile map, gemm1), gemm2 writing the
    routing-weighted bf16 partials ``[M, topk, H]``, aiter's top-k reduction.
    Returns ``[M, hidden_size]`` bf16."""
    from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

    from .gemm2 import gemm2_grid

    n_tokens = x.shape[0]
    topk = topk_ids.shape[1]
    device = x.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)
    bufs, _a_q, _a_s, h_q, h_s, num_m_blocks = a8w8_prefill_stage1(
        x,
        w13,
        w13_scale,
        topk_weights,
        topk_ids,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
    )
    partial = torch.empty(
        (n_tokens * topk, hidden_size), dtype=torch.bfloat16, device=device
    )
    num_m_blocks2 = (num_m_blocks * bm) // 128
    grid2 = gemm2_grid(num_m_blocks2, GEMM2_N_SPLIT)
    _run_compiled(
        _get_gemm2(hidden_size, intermediate_size, num_experts, topk, bm),
        h_q.view(-1),
        _u8_flat(w2),
        partial.view(-1),
        h_s,
        _u8_flat(w2_scale),
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks2,
        grid2,
        stream,
    )
    if out is None:
        out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    _run_moe_reduction(
        partial.view(n_tokens, topk, hidden_size), out, n_tokens, topk, hidden_size
    )
    return out


def a8w8_prefill_stage1(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
):
    """Sort + aiter's fp8 quant + tile map + gemm1. Returns ``(bufs, a_q, a_s,
    h_q, h_s, num_m_blocks)``: ``h_q`` fp8 ``[rows, I]`` and its e8m0 scales in
    sorted-row order (the layout gemm2 reads)."""
    from aiter import dtypes
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

    from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill.sort import SortBuffers
    from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill.tile_map import (
        tile_map_grid,
    )

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

    bufs = SortBuffers.allocate(n_tokens, num_experts, topk, bm, device)
    _get_sort(num_experts, topk, bm)(*bufs.launch_args(topk_ids, topk_weights, n_tokens))
    a_q, a_s = fused_dynamic_mx_quant_moe_sort(
        x,
        bufs.sorted_ids,
        bufs.num_valid_ids,
        token_num=n_tokens,
        topk=topk,
        block_size=bm,
        quant_dtype=dtypes.fp8,
    )
    num_m_blocks = bufs.max_sorted // bm
    rows = num_m_blocks * bm
    grid1 = tile_map_grid(num_m_blocks, inter)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=device)
    _run_compiled(
        _get_tile_map(inter, bm),
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        tile_map,
        grid1,
        stream,
    )
    h_q = torch.empty((rows, inter), dtype=torch.uint8, device=device)
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
        n_tokens,
        num_m_blocks,
        int(a_s.numel() * a_s.element_size()),
        tile_map,
        grid1,
        stream,
    )
    return bufs, a_q, a_s, h_q, h_s, num_m_blocks
