# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile cache + launch wrapper for the indexer kernels."""

import functools
import os

import torch

from . import score_decode, score_decode_v3, score_prefill, score_prefill_direct, score_prefill_w4b, score_prefill_w8, score_prefill_wp, score_prefill_dma
from .topk_prefill import NW as TOPK_ROWS_PER_WG
from .topk_prefill import TOPK, compile_topk_prefill
from .utils import _run_compiled

SPARSE_BLOCK_SIZE = 128
# Segment count: enough workgroups to fill the device several times over (the
# per-segment setup is one 128 KB query tile load), capped so a segment keeps a
# few score groups.
TARGET_WORKGROUPS = 1024


def _cdiv(a, b):
    return -(-a // b)


# lab switch between the 4-wave (512-row) and 8-wave (1024-row) scorers
SCORER = os.environ.get("M3_IDX_SCORER", "dma")
_scorer_mod = {"w4": score_prefill, "w4b": score_prefill_w4b, "w8": score_prefill_w8, "direct": score_prefill_direct, "wp": score_prefill_wp, "dma": score_prefill_dma}[SCORER]
TILE_Q, GROUP = _scorer_mod.TILE_Q, _scorer_mod.GROUP
MAX_SEGMENTS = 256 if SCORER == "w8" else 64


@functools.cache
def get_score_prefill():
    if SCORER == "w4":
        return score_prefill.compile_score_prefill()
    if SCORER == "w4b":
        return score_prefill_w4b.compile_score_prefill_w4b()
    if SCORER == "direct":
        return score_prefill_direct.compile_score_prefill_direct()
    if SCORER == "wp":
        return score_prefill_wp.compile_score_prefill_wp()
    if SCORER == "dma":
        return score_prefill_dma.compile_score_prefill_dma()
    return score_prefill_w8.compile_score_prefill_w8()


@functools.cache
def get_topk_prefill():
    return compile_topk_prefill()


def index_score_prefill(
    idx_q: torch.Tensor,  # [total_q, 1, 128] bf16
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, 128] bf16
    block_table: torch.Tensor,  # [batch, max_blocks] i32
    cu_seqlens_q: torch.Tensor,  # [batch + 1] i32
    seq_lens: torch.Tensor,  # [batch] i32
    prefix_lens: torch.Tensor,  # [batch] i32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int = 1,
) -> torch.Tensor:
    """Drop-in for ``minimax_m3_index_score`` (one local index head, bf16 cache):
    returns score ``[1, total_q, S]`` fp32, S = max_block rounded up to 16."""
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == 1 and num_kv_heads == 1 and head_dim == 128
    assert idx_q.dtype == torch.bfloat16 and index_kv_cache.dtype == torch.bfloat16
    assert idx_q.stride(2) == 1 and idx_q.stride(0) == head_dim
    assert index_kv_cache.is_contiguous() and tuple(index_kv_cache.shape[1:]) == (128, 128)
    kv_bytes = index_kv_cache.numel() * 2
    for t in (block_table, cu_seqlens_q, seq_lens, prefix_lens):
        assert t.dtype == torch.int32 and t.stride(-1) == 1
    batch = cu_seqlens_q.shape[0] - 1
    max_block = _cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    S = _cdiv(max_block, 16) * 16
    score = torch.empty((1, total_q, S), dtype=torch.float32, device=idx_q.device)
    qt = _cdiv(max_query_len, TILE_Q)
    nseg = max(
        1,
        min(MAX_SEGMENTS, TARGET_WORKGROUPS // max(1, qt * batch), max_block // GROUP),
    )
    _run_compiled(
        get_score_prefill(),
        idx_q.data_ptr(),
        index_kv_cache.data_ptr(),
        score.data_ptr(),
        block_table.data_ptr(),
        cu_seqlens_q.data_ptr(),
        seq_lens.data_ptr(),
        prefix_lens.data_ptr(),
        int(block_table.stride(0)),
        int(S),
        int(nseg),
        int(qt),
        int(total_q),
        int(kv_bytes),
        int(batch),
        int(_cdiv(qt * nseg * batch, 8) * 8),
        torch.cuda.current_stream(),
    )
    return score


def index_topk_prefill(
    score: torch.Tensor,  # [1, total_q, S] fp32
    cu_seqlens_q: torch.Tensor,  # [batch + 1] i32
    prefix_lens: torch.Tensor,  # [batch] i32
    max_query_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in for ``minimax_m3_index_topk`` (topk 16, one local head)."""
    assert topk == TOPK and score.shape[0] == 1 and score.dtype == torch.float32
    assert score.stride(2) == 1 and score.stride(1) == score.shape[2]
    total_q, S = score.shape[1], score.shape[2]
    batch = cu_seqlens_q.shape[0] - 1
    if out is not None:
        topk_idx = out[:, :total_q, :]
        assert topk_idx.is_contiguous()
    else:
        topk_idx = torch.empty((1, total_q, topk), dtype=torch.int32, device=score.device)
    qbs = _cdiv(max_query_len, TOPK_ROWS_PER_WG)
    _run_compiled(
        get_topk_prefill(),
        score.data_ptr(),
        topk_idx.data_ptr(),
        cu_seqlens_q.data_ptr(),
        prefix_lens.data_ptr(),
        int(S),
        int(qbs),
        int(total_q),
        int(init_blocks),
        int(local_blocks),
        int(qbs * batch),
        torch.cuda.current_stream(),
    )
    return topk_idx


DECODE = os.environ.get("M3_IDX_DECODE", "v3")  # lab switch


@functools.cache
def get_score_decode():
    if DECODE == "v1":
        return score_decode.compile_score_decode()
    return score_decode_v3.compile_score_decode_v3()


def index_score_decode(
    idx_q: torch.Tensor,  # [num_reqs * decode_query_len, 1, 128] bf16
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,  # [num_reqs, max_blocks] i32
    seq_lens: torch.Tensor,  # [num_reqs] i32
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    decode_query_len: int,
) -> torch.Tensor:
    """The score phase of ``minimax_m3_index_decode``: ``[1, total_q, S]`` fp32,
    S = max_block rounded up to 16 (the fused Triton selector's input)."""
    total_q, num_idx_heads, head_dim = idx_q.shape
    num_reqs = seq_lens.shape[0]
    assert num_idx_heads == 1 and head_dim == 128 and total_q == num_reqs * decode_query_len
    assert 1 <= decode_query_len <= 16 and num_reqs <= score_decode.MAX_REQS, num_reqs
    assert idx_q.dtype == torch.bfloat16 and index_kv_cache.dtype == torch.bfloat16
    assert idx_q.stride(2) == 1 and idx_q.stride(0) == head_dim
    assert index_kv_cache.is_contiguous() and tuple(index_kv_cache.shape[1:]) == (128, 128)
    kv_bytes = index_kv_cache.numel() * 2
    assert block_table.dtype == torch.int32 and seq_lens.dtype == torch.int32
    assert block_table.stride(1) == 1 and seq_lens.is_contiguous()
    max_block = _cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    S = _cdiv(max_block, 16) * 16
    score = torch.empty((1, total_q, S), dtype=torch.float32, device=idx_q.device)
    _run_compiled(
        get_score_decode(),
        idx_q.data_ptr(),
        index_kv_cache.data_ptr(),
        score.data_ptr(),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(block_table.stride(0)),
        int(S),
        int(num_reqs),
        int(decode_query_len),
        int(init_blocks),
        int(local_blocks),
        int(total_q),
        int(kv_bytes),
        torch.cuda.current_stream(),
    )
    return score


def index_decode(
    idx_q, index_kv_cache, block_table, seq_lens, max_seq_len, topk, init_blocks,
    local_blocks, num_kv_heads, decode_query_len, max_decode_query_len, out=None, *,
    attention_block_table=None, sparse_block_table_out=None, sparse_context_lens_out=None,
    block_page_stride=None, completion_counter=None,
):
    """Drop-in for ``minimax_m3_index_decode``: the FlyDSL scorer followed by the
    Triton fused top-k / sparse-table selector (its launch code, unchanged)."""
    from vllm.models.minimax_m3.amd.ops import index_topk as up
    from vllm.models.minimax_m3.amd.ops.sparse_pa import PAGES_PER_SPARSE_BLOCK
    from vllm.platforms.rocm import on_gfx950
    from vllm.triton_utils import triton

    assert num_kv_heads == 1
    num_idx_heads = 1
    total_q = idx_q.shape[0]
    batch = total_q
    score = index_score_decode(
        idx_q, index_kv_cache, block_table, seq_lens, max_seq_len, init_blocks,
        local_blocks, decode_query_len,
    )
    max_block = _cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    emit_sparse_table = attention_block_table is not None
    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty((1, total_q, topk), dtype=torch.int32, device=idx_q.device)
    num_topk_chunks, single_tile_guaranteed, adaptive_final_merge = up._decode_topk_launch_policy(
        max_block, batch, num_idx_heads, topk, is_gfx950=on_gfx950()
    )
    block_size_t = triton.next_power_of_2(topk)
    topk_partial = torch.empty(
        num_topk_chunks, num_idx_heads, batch, block_size_t, dtype=torch.int64, device=idx_q.device
    )
    if completion_counter is None:
        active_counter = torch.zeros((num_idx_heads, batch), dtype=torch.int32, device=idx_q.device)
    else:
        active_counter = completion_counter[:, :batch]
    sel_bt, sel_sbt, sel_ctx = block_table, topk_idx, seq_lens
    sel_stride, sel_sbt_stride = PAGES_PER_SPARSE_BLOCK, topk_idx.stride(1)
    if emit_sparse_table:
        sel_bt, sel_sbt, sel_ctx = attention_block_table, sparse_block_table_out, sparse_context_lens_out
        sel_stride, sel_sbt_stride = block_page_stride, sparse_block_table_out.stride(0)
    up._decode_topk_fused_kernel[(batch, num_idx_heads, num_topk_chunks)](
        score, topk_partial, active_counter, topk_idx, seq_lens, sel_bt, sel_sbt, sel_ctx,
        decode_query_len,
        score.stride(0), score.stride(1), score.stride(2),
        topk_partial.stride(0), topk_partial.stride(1), topk_partial.stride(2), topk_partial.stride(3),
        active_counter.stride(0), active_counter.stride(1),
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        sel_bt.stride(0), sel_sbt_stride,
        topk=topk, block_size=SPARSE_BLOCK_SIZE, pages_per_sparse_block=PAGES_PER_SPARSE_BLOCK,
        block_page_stride=sel_stride, NUM_TOPK_CHUNKS=num_topk_chunks, BLOCK_SIZE_K=512,
        BLOCK_SIZE_T=block_size_t, EMIT_SPARSE_TABLE=emit_sparse_table,
        SINGLE_TILE_GUARANTEED=single_tile_guaranteed, ADAPTIVE_FINAL_MERGE=adaptive_final_merge,
        num_warps=4 if adaptive_final_merge else 8, num_stages=2,
    )
    return topk_idx
