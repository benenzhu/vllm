# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrappers for the fp8 indexer kernels, with the AITER
``pa_sparse_block_score_*`` signatures ``indexer_aiter.py`` calls."""

import functools

import torch

from . import score_prefill as _score_prefill
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
