# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrappers for the fp8 indexer kernels, with the AITER
``pa_sparse_block_score_*`` signatures ``indexer_aiter.py`` calls."""

import functools

import torch

from . import score_decode as _score_decode
from . import score_prefill as _score_prefill
from . import topk as _topk
from .utils import _run_compiled

SPARSE_BLOCK_SIZE = 128
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
# Segment count: enough workgroups to fill the device several times over (the
# per-segment setup is one 64 KB query tile load), capped so a segment keeps a
# few score groups.
TARGET_WORKGROUPS = 1024
MAX_SEGMENTS = 64


def _cdiv(a, b):
    return -(-a // b)


@functools.cache
def get_score_prefill():
    return _score_prefill.compile_score_prefill()


def score_prefill(
    q_idx: torch.Tensor,  # [total_q, 1, 128] e4m3
    key_cache_idx: torch.Tensor,  # [num_blocks, 128, 128] e4m3
    score: torch.Tensor,  # [1, total_q, S] fp32, written in place
    block_table: torch.Tensor,  # [batch, >= max_block] i32
    cu_seqlens_q: torch.Tensor,  # [batch + 1] i32
    seq_lens: torch.Tensor,  # [batch] i32
    init_blocks: int = 0,
    local_blocks: int = 0,
    max_query_len: int = 0,
    max_seq_len: int = 0,
) -> None:
    """Drop-in for AITER's ``pa_sparse_block_score_prefill`` (one local index
    head): score[0, n, b] = max over the causal tokens of block b of q[n] . k,
    init / local blocks carry the 1e30 / 1e29 sentinels, blocks a row cannot
    see are left as they are."""
    total_q, num_idx_heads, head_dim = q_idx.shape
    assert num_idx_heads == 1 and head_dim == 128
    assert q_idx.dtype in FP8_DTYPES and key_cache_idx.dtype == q_idx.dtype
    assert q_idx.is_contiguous() and key_cache_idx.is_contiguous()
    assert tuple(key_cache_idx.shape[1:]) == (SPARSE_BLOCK_SIZE, head_dim)
    assert score.dtype == torch.float32 and score.stride(2) == 1
    assert score.shape[0] == 1 and score.shape[1] == total_q and score.stride(1) == score.shape[2]
    for t in (block_table, cu_seqlens_q, seq_lens):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    assert max_query_len >= 1 and max_seq_len >= 1
    batch = cu_seqlens_q.shape[0] - 1
    if batch == 0 or total_q == 0:
        return
    S = score.shape[2]
    max_block = _cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    assert S >= max_block
    qt = _cdiv(max_query_len, _score_prefill.TILE_Q)
    nseg = max(
        1,
        min(MAX_SEGMENTS, TARGET_WORKGROUPS // max(1, qt * batch), max_block // _score_prefill.GROUP),
    )
    _run_compiled(
        get_score_prefill(),
        q_idx.data_ptr(),
        key_cache_idx.data_ptr(),
        score.data_ptr(),
        block_table.data_ptr(),
        cu_seqlens_q.data_ptr(),
        seq_lens.data_ptr(),
        int(block_table.stride(0)),
        int(S),
        int(nseg),
        int(qt),
        int(total_q),
        int(batch),
        int(((batch - 1) * block_table.stride(0) + block_table.shape[1]) * 4),
        int(init_blocks),
        int(local_blocks),
        int(_cdiv(qt * nseg * batch, 8) * 8),
        torch.cuda.current_stream(),
    )


@functools.cache
def get_score_decode():
    return _score_decode.compile_score_decode()


def score_decode(
    q_idx: torch.Tensor,  # [num_reqs * query_len, 1, 128] e4m3
    key_cache_idx: torch.Tensor,  # [num_blocks, 128, 128] e4m3
    score: torch.Tensor,  # [1, num_reqs * query_len, S] fp32, written in place
    block_table: torch.Tensor,  # [num_reqs, >= max_block] i32
    seq_lens: torch.Tensor,  # [num_reqs] i32
    init_blocks: int = 0,
    local_blocks: int = 0,
    query_len: int = 1,
    max_seq_len: int = 0,
) -> None:
    """Drop-in for AITER's ``pa_sparse_block_score_decode`` (one local index
    head, uniform query length ≤ 16). The kernel's work split is one lane per
    request, so requests are scored in groups of 64 (one launch each)."""
    total_q, num_idx_heads, head_dim = q_idx.shape
    num_reqs = seq_lens.shape[0]
    assert num_idx_heads == 1 and head_dim == 128 and total_q == num_reqs * query_len
    assert 1 <= query_len <= 16
    assert q_idx.dtype in FP8_DTYPES and key_cache_idx.dtype == q_idx.dtype
    assert q_idx.is_contiguous() and key_cache_idx.is_contiguous()
    assert tuple(key_cache_idx.shape[1:]) == (SPARSE_BLOCK_SIZE, head_dim)
    assert score.dtype == torch.float32 and score.stride(2) == 1
    assert score.shape[0] == 1 and score.shape[1] == total_q and score.stride(1) == score.shape[2]
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert seq_lens.dtype == torch.int32 and seq_lens.stride(0) == 1
    if num_reqs == 0:
        return
    S = score.shape[2]
    assert S >= _cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    for r0 in range(0, num_reqs, _score_decode.MAX_REQS):
        n = min(_score_decode.MAX_REQS, num_reqs - r0)
        _run_compiled(
            get_score_decode(),
            q_idx.data_ptr() + r0 * query_len * q_idx.stride(0) * q_idx.element_size(),
            key_cache_idx.data_ptr(),
            score.data_ptr() + r0 * query_len * score.stride(1) * 4,
            block_table.data_ptr() + r0 * block_table.stride(0) * 4,
            seq_lens.data_ptr() + r0 * 4,
            int(block_table.stride(0)),
            int(S),
            int(n),
            int(query_len),
            int(init_blocks),
            int(local_blocks),
            int(n * query_len),
            torch.cuda.current_stream(),
        )


@functools.cache
def get_topk():
    return _topk.compile_topk()


def topk(
    score: torch.Tensor,  # [1, total_q, S] fp32
    topk_idx: torch.Tensor,  # [1, total_q, 16] i32, written in place (may be a row slice)
    block_table: torch.Tensor,  # the attend's (page-16) block table [num_reqs, stride] i32
    seq_lens: torch.Tensor,  # [num_reqs] i32 (uniform rows)
    sparse_bt: torch.Tensor,  # [total_q * num_kv_heads, 16 * 8] i32, written in place
    sparse_ctx: torch.Tensor,  # [total_q * num_kv_heads] i32, written in place
    max_seq_len: int = 0,
    block_size: int = 0,
    query_len: int = 1,
    num_waves: int = 0,
    num_valid_pages: torch.Tensor | None = None,
    row_req_id: torch.Tensor | None = None,
    kv_lens: torch.Tensor | None = None,
    num_kv_heads: int = 1,
    pages_per_block: int = 8,
) -> None:
    """Drop-in for AITER's ``pa_sparse_block_topk`` (one local head, topk 16,
    8 pages per block): the 16 best blocks of every row (-1 padded) and the
    attend's page table + token count per (row, kv head). Ragged rows come with
    ``num_valid_pages`` / ``row_req_id`` / ``kv_lens``; uniform rows use
    ``query_len`` and ``seq_lens``."""
    num_idx_heads, total_q, S = score.shape
    assert num_idx_heads == 1 and score.dtype == torch.float32 and score.stride(2) == 1
    assert score.stride(1) == S
    assert topk_idx.dtype == torch.int32 and topk_idx.shape[0] == 1 and topk_idx.shape[1] == total_q
    assert topk_idx.shape[2] == _topk.TOPK and topk_idx.stride(2) == 1
    assert block_size == _topk.BLK and pages_per_block == _topk.PPB
    assert num_kv_heads >= 1
    rows = total_q * num_kv_heads
    assert sparse_bt.dtype == torch.int32 and sparse_bt.shape == (rows, _topk.TOPK * _topk.PPB)
    assert sparse_bt.stride(1) == 1 and sparse_ctx.dtype == torch.int32 and sparse_ctx.numel() == rows
    assert sparse_ctx.stride(0) == 1 and block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert S >= _cdiv(max_seq_len, block_size)
    if num_valid_pages is not None:
        assert row_req_id is not None and kv_lens is not None
        for t in (row_req_id, kv_lens):
            assert t.dtype == torch.int32 and t.numel() == total_q and t.stride(0) == 1
        qlen, rid, kvl, rows_bytes = 0, row_req_id.data_ptr(), kv_lens.data_ptr(), total_q * 4
    else:
        assert query_len >= 1 and total_q % query_len == 0
        assert seq_lens.dtype == torch.int32 and seq_lens.stride(0) == 1
        assert seq_lens.numel() == total_q // query_len
        qlen, rid, kvl, rows_bytes = query_len, 0, 0, 0
    if total_q == 0:
        return
    _run_compiled(
        get_topk(),
        score.data_ptr(),
        topk_idx.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        rid,
        kvl,
        sparse_bt.data_ptr(),
        sparse_ctx.data_ptr(),
        int(seq_lens.numel() * 4),
        int(rows_bytes),
        int(S),
        int(total_q),
        int(qlen),
        int(topk_idx.stride(1)),
        int(block_table.stride(0)),
        int(sparse_bt.stride(0)),
        int(num_kv_heads),
        int(_cdiv(total_q, _topk.NW)),
        torch.cuda.current_stream(),
    )
