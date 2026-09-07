# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrappers for the decode gemm1 / gemm2 kernels.

Sorted-path contract (aiter ``moe_sorting``): ``sorted_token_ids`` (token in the low
24 bits, topk slot in the high 8), ``sorted_expert_ids`` (one per 16-row block),
``sorted_weights``, ``num_valid_ids`` (int32[1]: padded sorted row count) and a
zeroed output. ``inline_sort``: no sort buffers, ``topk_ids`` / ``topk_weights``
directly.
"""

import functools

import flydsl.compiler as flyc
import torch

from .gemm1 import BM, compile_gemm1
from .gemm2 import compile_gemm2


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction (the shim aiter ships in ``ops/flydsl/kernels``)."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


_launches: dict[str, object] = {}


def _get(compile_fn, **kw):
    """One launch object per distinct kernel. The kernels depend on ``n_tokens``
    only through a few buckets, so launches built for different ``n_tokens`` are
    shared by kernel name (the compiled code hangs off the launch object)."""
    launch = compile_fn(**kw)
    return _launches.setdefault(launch.kernel_name, launch)


@functools.cache
def get_gemm1(**kw):
    return _get(compile_gemm1, **kw)


@functools.cache
def get_gemm2(**kw):
    return _get(compile_gemm2, **kw)


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
    alpha=1.702,
    swiglu_limit=7.0,
    w_layout="standard",
    sorted_expert_ids=None,
    num_valid_ids=None,
    sorted_token_ids=None,
    inline_sort=False,
    topk_ids=None,
    zero_out=None,
    stream=None,
):
    """Stage 1: gate/up GEMM + swiglu-OAI -> bf16 ``[sorted rows, D_INTER]``."""
    launch = get_gemm1(
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        n_tokens=int(n_tokens),
        w_layout=w_layout,
        inline_sort=inline_sort,
    )
    if inline_sort:
        assert int(n_tokens) <= BM and topk_ids is not None and zero_out is not None
        max_m_blocks = int(n_tokens) * int(topk)
        eids_ptr, cumsum_ptr, mind_ptr = 0, 0, topk_ids.data_ptr()
        zero_ptr = zero_out.data_ptr()
        zero_dw = (zero_out.numel() * zero_out.element_size()) // 4
    else:
        max_m_blocks = int(sorted_expert_ids.numel())
        eids_ptr, cumsum_ptr, mind_ptr = (
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            sorted_token_ids.data_ptr(),
        )
        zero_ptr, zero_dw = 0, 0
    grid = max_m_blocks * (D_INTER // launch.tile_n)
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
    sorted_expert_ids=None,
    num_valid_ids=None,
    sorted_token_ids=None,
    sorted_weights=None,
    inline_sort=False,
    topk=None,
    topk_ids=None,
    topk_weights=None,
    stream=None,
):
    """Stage 2: down GEMM, routing-weighted bf16 atomic add into ``out_bf16``
    ``[n_tokens, D_HIDDEN]`` (zeroed beforehand)."""
    launch = get_gemm2(
        NE=NE,
        N_OUT=D_HIDDEN,
        D_INTER=D_INTER,
        n_tokens=int(n_tokens),
        inline_sort=inline_sort,
        TOPK=topk if inline_sort else None,
    )
    if inline_sort:
        assert int(n_tokens) <= BM and topk_ids is not None and topk_weights is not None
        assert topk_weights.dtype == torch.float32 and topk_weights.is_contiguous()
        max_m_blocks = int(n_tokens) * int(topk)
        eids_ptr, cumsum_ptr = 0, 0
        stids_ptr, sw_ptr = topk_ids.data_ptr(), topk_weights.data_ptr()
    else:
        max_m_blocks = int(sorted_expert_ids.numel())
        eids_ptr, cumsum_ptr = sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr()
        stids_ptr, sw_ptr = sorted_token_ids.data_ptr(), sorted_weights.data_ptr()
    grid = max_m_blocks * (D_HIDDEN // launch.tile_n) * launch.ksplit
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
        int(grid),
        out_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return out_bf16
