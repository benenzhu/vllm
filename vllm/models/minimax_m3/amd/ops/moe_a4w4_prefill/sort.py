# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 prefill routing sort: FlyDSL port of aiter's 3-stage mxfp4 sorter
(``csrc/kernels/mxfp4_moe/moe_aux/moe_3stage_sort.cuh``), token count a runtime
argument. Three launches per call:

  sort_count      grid SORT_CTAS  each CTA histograms its slice of the (token,
                                  slot) pairs in LDS and writes
                                  ``block_offsets[e, cta] = count``
  sort_cumsum     grid 1          per expert: total, padded to block_m; a serial
                                  scan gives the expert start rows;
                                  ``block_offsets[e, cta]`` becomes the per-CTA
                                  start row; ``sorted_expert_ids`` per block;
                                  ``num_valid_ids[0]`` = padded total
  sort_place_pad  grid SORT_CTAS  each CTA places its pairs (LDS atomic on its
                                  per-expert cursor) and pads its share of experts

Outputs are aiter's layout: ``sorted_ids[row] = token | slot << 24`` (padding
rows: ``n_tokens``), ``sorted_weights`` (padding 0), ``sorted_expert_ids`` per
block, ``num_valid_ids[0]``. Rows inside one expert come out in arrival order
of the atomics, i.e. not deterministic (same as the HIP kernel); the consumers
do not depend on it.
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr

SORT_CTAS = 128  # histogram / placement blocks (place_pad at 32768 tokens: 18 us with 32, 6 with 128)
THREADS = 1024  # >= E and >= block_m


def max_sorted_rows(n_tokens: int, E: int, topk: int, block_m: int) -> int:
    """aiter's bound: every active expert pads by < block_m rows."""
    active = min(E, n_tokens * topk)
    cumsum_max = n_tokens * topk + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


@dataclass
class SortBuffers:
    sorted_ids: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor  # [2] i32, [0] = padded row count
    sorted_weights: torch.Tensor
    block_offsets: torch.Tensor  # workspace [E * SORT_CTAS]
    real_counts: torch.Tensor  # workspace [E]
    max_sorted: int
    block_m: int

    @staticmethod
    def allocate(n_tokens, E, topk, block_m, device):
        ms = max_sorted_rows(n_tokens, E, topk, block_m)
        i32 = torch.int32
        return SortBuffers(
            sorted_ids=torch.empty(ms, dtype=i32, device=device),
            sorted_expert_ids=torch.empty(ms // block_m, dtype=i32, device=device),
            num_valid_ids=torch.empty(2, dtype=i32, device=device),  # written by sort_cumsum
            sorted_weights=torch.empty(ms, dtype=torch.float32, device=device),
            block_offsets=torch.empty(E * SORT_CTAS, dtype=i32, device=device),
            real_counts=torch.empty(E, dtype=i32, device=device),
            max_sorted=ms,
            block_m=block_m,
        )

    def launch_args(self, topk_ids, topk_weights, n_tokens):
        return (
            topk_ids.contiguous().int().view(-1),
            topk_weights.contiguous().float().view(-1),
            self.sorted_ids,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.sorted_weights,
            self.block_offsets,
            self.real_counts,
            int(n_tokens),
            torch.cuda.current_stream(),
        )


def _lds_atomic_add_i32(ptr, value):
    """``old = *ptr; *ptr += value`` on an LDS i32 pointer; returns ``old`` (fx has
    no atomic wrapper, so this is the one raw LLVM op)."""
    return fx.Int32(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            ptr.llvm_ptr,
            fx.Int32(value).ir_value(),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


@functools.cache
def compile_moe_sort(*, E: int, topk: int, block_m: int):
    assert (block_m & (block_m - 1)) == 0 and max(E, block_m) <= THREADS
    experts_per_cta = (E + SORT_CTAS - 1) // SORT_CTAS
    bm_shift = block_m.bit_length() - 1
    tag = f"E{E}_K{topk}_BM{block_m}"

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, E]  # per-CTA count / expert total / cursor
        padded: fx.Array[fx.Int32, E]
        starts: fx.Array[fx.Int32, E + 1]  # expert start rows, [E] = padded total

    def pair_range(n_tok, bx):
        """this CTA's [start, end) of the n_tok * topk routing pairs"""
        total = n_tok * topk
        per_cta = (total + (SORT_CTAS - 1)) // SORT_CTAS
        start = bx * per_cta
        end = start + per_cta
        return start, (end < total).select(end, total)

    @flyc.kernel(name=f"m3_sort_count_{tag}", known_block_size=[THREADS, 1, 1])
    def sort_count(topk_ids: fx.Tensor, block_offsets: fx.Tensor, n_tok: fx.Int32):
        count = fx.SharedAllocator().allocate(Shared).peek().count.ptr
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        start, end = pair_range(n_tok, bx)
        if tx < E:
            count[tx] = 0
        fx.gpu.barrier()
        for i in range(start + tx, end, THREADS):
            _lds_atomic_add_i32(count + fx.Int32(topk_ids[fx.Int32(i)]), 1)
        fx.gpu.barrier()
        if tx < E:
            block_offsets[tx * SORT_CTAS + bx] = fx.Int32(count[tx])

    @flyc.kernel(name=f"m3_sort_cumsum_{tag}", known_block_size=[THREADS, 1, 1])
    def sort_cumsum(
        block_offsets: fx.Tensor,
        real_counts: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        total_c, padded_c, starts = smem.count.ptr, smem.padded.ptr, smem.starts.ptr
        tx = fx.thread_idx.x
        if tx < E:
            total = fx.Int32(0)
            for c in range_constexpr(SORT_CTAS):
                total = total + fx.Int32(block_offsets[tx * SORT_CTAS + c])
            total_c[tx] = total
            padded_c[tx] = (total + (block_m - 1)) & ~(block_m - 1)
            real_counts[tx] = total
        fx.gpu.barrier()
        if tx == 0:
            acc = fx.Int32(0)  # serial scan over the E padded counts
            for e in range_constexpr(E):
                starts[e] = acc
                acc = acc + fx.Int32(padded_c[e])
            starts[E] = acc
            num_valid[0] = acc
        fx.gpu.barrier()
        if tx < E:
            # block_offsets[e, cta] -> that CTA's first row of expert e
            acc = fx.Int32(starts[tx])
            for c in range_constexpr(SORT_CTAS):
                off = tx * SORT_CTAS + c
                cnt = fx.Int32(block_offsets[off])
                block_offsets[off] = acc
                acc = acc + cnt
            for b in range(
                fx.Int32(starts[tx]) >> bm_shift, fx.Int32(starts[tx + 1]) >> bm_shift
            ):
                sorted_expert_ids[fx.Int32(b)] = tx

    @flyc.kernel(name=f"m3_sort_place_pad_{tag}", known_block_size=[THREADS, 1, 1])
    def sort_place_pad(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        block_offsets: fx.Tensor,
        real_counts: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        n_tok: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        cursor, starts = smem.count.ptr, smem.starts.ptr
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        if tx < E:
            cursor[tx] = fx.Int32(block_offsets[tx * SORT_CTAS + bx])
            starts[tx] = fx.Int32(block_offsets[tx * SORT_CTAS])
        if tx == 0:
            starts[E] = fx.Int32(num_valid[0])
        fx.gpu.barrier()
        start, end = pair_range(n_tok, bx)
        for i in range(start + tx, end, THREADS):
            p = fx.Int32(i)
            row = _lds_atomic_add_i32(cursor + fx.Int32(topk_ids[p]), 1)
            sorted_ids[row] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
            sorted_w[row] = fx.Float32(topk_w[p])
        fx.gpu.barrier()
        # padding: every expert pads by < block_m rows, lanes [0, block_m) do it
        if tx < block_m:
            for ee in range_constexpr(experts_per_cta):
                e = bx * experts_per_cta + ee
                if e < E:
                    lo = fx.Int32(starts[e]) + fx.Int32(real_counts[e]) + tx
                    for r in range(lo, fx.Int32(starts[e + 1]), THREADS):
                        sorted_ids[fx.Int32(r)] = n_tok  # token n_tokens, slot 0
                        sorted_w[fx.Int32(r)] = fx.Float32(0.0)

    @flyc.jit
    def launch_sort(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid: fx.Tensor,
        sorted_w: fx.Tensor,
        block_offsets: fx.Tensor,
        real_counts: fx.Tensor,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        sort_count(topk_ids, block_offsets, n_tokens).launch(
            grid=(SORT_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream
        )
        sort_cumsum(block_offsets, real_counts, num_valid, sorted_expert_ids).launch(
            grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream
        )
        sort_place_pad(
            topk_ids,
            topk_w,
            block_offsets,
            real_counts,
            num_valid,
            sorted_ids,
            sorted_w,
            n_tokens,
        ).launch(grid=(SORT_CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    return launch_sort
