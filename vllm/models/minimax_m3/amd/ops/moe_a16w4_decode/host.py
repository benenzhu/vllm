# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host glue: compile cache + launch wrappers for the a16w4 gemm1/gemm2 kernels.

Slimmed from FlyDSL ``tests/kernels/moe_a16wmix_host.py``: no CSV config lookup, every
tile parameter is explicit so sweeps are reproducible from the command line.

Sorting contract (aiter ``moe_sorting``): ``sorted_token_ids`` (token in low 24 bits,
topk slot in high 8), ``sorted_expert_ids`` (one per BM-row block), ``sorted_weights``,
``num_valid_ids`` (int32[1]: padded sorted row count) and ``moe_buf`` (zeroed output).
"""

import functools

import flydsl.compiler as flyc
import torch

from .gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the cached
    CompiledFunction. Same shim aiter ships in ``ops/flydsl/kernels/tensor_shim.py``."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def get_gemm1(**kw):
    return compile_gemm1_a16w4_port(**kw)


@functools.cache
def get_gemm2(**kw):
    return compile_gemm2_a16w4_port(**kw)


def _pairs_cap(n_tokens, topk, tile_m):
    """Routing pairs the pairs-mode table scans (64 per wave pass): one pass up to
    M*topk
    <= 64 (M <= 12 at topk 5), else the full BM*topk (80 -> two passes, M = 16)."""
    return 64 if int(n_tokens) * int(topk) <= 64 else int(tile_m) * int(topk)


def a16w4_gemm1(
    *,
    x_bf16,
    w1_u8,
    w1_scale_u8,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m,
    sorted_expert_ids=None,
    num_valid_ids=None,
    sorted_token_ids=None,
    tile_n,
    tile_k,
    k_wave=1,
    b_nt=0,
    xcd_swizzle=0,
    waves_per_eu=None,
    act="swigluoai",
    alpha=1.702,
    swiglu_limit=7.0,
    w_layout="standard",
    a_direct=False,
    prefetch=1,
    scale_share=False,
    pairs=False,
    topk_ids=None,
    zero_out=None,
    a_rows4=False,
    stream=None,
):
    """Stage 1: gate/up GEMM + activation -> bf16 ``[sorted_size, D_INTER]`` by sorted
    row."""
    launch = get_gemm1(
        BM=tile_m,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        TILE_N=tile_n,
        TILE_K=tile_k,
        act=act,
        b_cache_mod=b_nt,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype="mxfp4",
        w_layout=w_layout,
        k_wave=k_wave,
        a_direct=a_direct,
        prefetch=prefetch,
        scale_share=scale_share,
        pairs=pairs,
        max_pairs=_pairs_cap(n_tokens, topk, tile_m) if pairs else None,
        a_rows4=a_rows4,
    )
    if pairs:
        # sort-free decode routing: one m-block per routing pair, the kernel finds its
        # rows
        assert int(n_tokens) <= tile_m, "pairs mode needs n_tokens <= tile_m"
        assert topk_ids is not None and zero_out is not None
        max_m_blocks = int(n_tokens) * int(topk)
        eids_ptr, cumsum_ptr, mind_ptr = 0, 0, topk_ids.data_ptr()
        zero_ptr, zero_dw = (
            zero_out.data_ptr(),
            (zero_out.numel() * zero_out.element_size()) // 4,
        )
    else:
        max_m_blocks = int(sorted_expert_ids.numel())
        eids_ptr, cumsum_ptr, mind_ptr = (
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            sorted_token_ids.data_ptr(),
        )
        zero_ptr, zero_dw = 0, 0
    grid = gemm1_a16w4_grid(
        tile_m, INTER=D_INTER, TILE_N=tile_n, max_m_blocks=max_m_blocks
    )
    # f32 launch slots: situ_beta, situ_beta_rcp, situ_linbeta, situ_linbeta_rcp,
    # swiglu_limit.
    # swigluoai reads alpha from the situ_beta slot and the clamp bound from
    # swiglu_limit.
    _run_compiled(
        launch,
        x_bf16.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        eids_ptr,
        cumsum_ptr,
        mind_ptr,
        int(n_tokens),
        int(grid),
        float(alpha),
        1.0 / float(alpha),
        1.0,
        1.0,
        float(swiglu_limit),
        inter_sorted_bf16.data_ptr(),
        int(zero_ptr),
        int(zero_dw),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16


def a16w4_gemm2(
    *,
    inter_sorted_bf16,
    w2_u8,
    w2_scale_u8,
    out_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    tile_m,
    tile_n,
    tile_k,
    sorted_expert_ids=None,
    num_valid_ids=None,
    sorted_token_ids=None,
    sorted_weights=None,
    b_nt=0,
    xcd_swizzle=1,
    waves_per_eu=None,
    persist=False,
    a_direct=False,
    prefetch=1,
    ksplit=1,
    pad_mask=False,
    hoist=None,
    pairs=False,
    topk=None,
    topk_ids=None,
    topk_weights=None,
    scale_share=False,
    stream=None,
):
    """Stage 2: down GEMM, routing-weighted bf16 atomic add into ``out_bf16``
    [n_tokens, D_HIDDEN]."""
    launch = get_gemm2(
        BM=tile_m,
        NE=NE,
        N_OUT=D_HIDDEN,
        D_INTER=D_INTER,
        TILE_N=tile_n,
        TILE_K=tile_k,
        b_cache_mod=b_nt,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype="mxfp4",
        persist=persist,
        a_direct=a_direct,
        prefetch=prefetch,
        ksplit=ksplit,
        pad_mask=pad_mask,
        hoist=hoist,
        pairs=pairs,
        TOPK=topk if pairs else None,
        scale_share=scale_share,
        max_pairs=_pairs_cap(n_tokens, topk, tile_m) if pairs else None,
    )
    if pairs:
        assert (
            int(n_tokens) <= tile_m
            and topk_ids is not None
            and topk_weights is not None
        )
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        max_m_blocks = int(n_tokens) * int(topk)
        eids_ptr, cumsum_ptr = 0, 0
        stids_ptr, sw_ptr = topk_ids.data_ptr(), topk_weights.data_ptr()
    else:
        max_m_blocks = int(sorted_expert_ids.numel())
        eids_ptr, cumsum_ptr = sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr()
        stids_ptr, sw_ptr = sorted_token_ids.data_ptr(), sorted_weights.data_ptr()
    grid = gemm2_a16w4_grid(
        tile_m,
        N_OUT=D_HIDDEN,
        TILE_N=tile_n,
        max_m_blocks=max_m_blocks,
        persist=persist,
        ksplit=ksplit,
    )
    _run_compiled(
        launch,
        inter_sorted_bf16.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        eids_ptr,
        cumsum_ptr,
        stids_ptr,
        sw_ptr,
        int(n_tokens),
        int(max_m_blocks),
        int(grid),
        out_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out_bf16
