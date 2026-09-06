# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MiniMax-M3 MoE routing sort: FlyDSL port of aiter's 3-stage mxfp4 sorter
(``csrc/kernels/mxfp4_moe/moe_aux/moe_3stage_sort.cuh``, the one the Kimi
FlyDSL migration re-implemented), generalised to any (E, topk, block_m) with
the token count a runtime argument.

Three launches per call:

  sort_count     grid = sort_ctas   each CTA histograms its slice of the
                                    (token, slot) pairs into LDS (atomics) and
                                    writes ``block_offsets[e, cta] = count``
  sort_cumsum    grid = 1           per expert: total, padded-to-block_m; a
                                    serial scan gives the expert start rows;
                                    ``block_offsets[e, cta]`` becomes the
                                    per-CTA start row; ``sorted_expert_ids``
                                    filled per block; ``cumsum[0]`` = padded
                                    total (= aiter ``num_valid_ids[0]``)
  sort_place_pad grid = sort_ctas   each CTA places its pairs (LDS atomic on
                                    its per-expert cursor) and pads its share
                                    of experts with ``n_tokens`` sentinels

Outputs are aiter's layout: ``sorted_ids[row] = token | slot << 24`` (padding
rows: ``n_tokens``), ``sorted_weights`` (padding 0), ``sorted_expert_ids`` per
block, plus ``m_indices`` (plain token per row), ``reverse_sorted[pair] = row``
(for a scatter-reduce after gemm2) and ``masked_m[e]`` (padded rows of e).
Rows inside one expert come out in arrival order of the atomics, i.e. not
deterministic (same as the HIP kernel); consumers must not depend on it.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import memref, scf, vector
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .lowlevel import (
    extract_global_ptr,
    global_load_f32,
    global_load_i32,
    global_load_i32_vec,
    global_store_f32,
    global_store_i32,
    lds_atomic_add_i32,
    ptr_arg,
)


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
    reverse_sorted: torch.Tensor
    sorted_weights: torch.Tensor
    masked_m: torch.Tensor
    m_indices: torch.Tensor
    block_offsets: torch.Tensor  # workspace [E * sort_ctas]
    real_counts: torch.Tensor  # workspace [E]
    max_sorted: int
    block_m: int

    @staticmethod
    def allocate(n_tokens, E, topk, block_m, sort_ctas, device):
        ms = max_sorted_rows(n_tokens, E, topk, block_m)
        i32 = torch.int32
        return SortBuffers(
            sorted_ids=torch.empty(ms, dtype=i32, device=device),
            sorted_expert_ids=torch.empty(ms // block_m, dtype=i32, device=device),
            num_valid_ids=torch.zeros(2, dtype=i32, device=device),
            reverse_sorted=torch.empty(n_tokens * topk, dtype=i32, device=device),
            sorted_weights=torch.empty(ms, dtype=torch.float32, device=device),
            masked_m=torch.empty(E, dtype=i32, device=device),
            m_indices=torch.empty(ms, dtype=i32, device=device),
            block_offsets=torch.empty(E * sort_ctas, dtype=i32, device=device),
            real_counts=torch.empty(E, dtype=i32, device=device),
            max_sorted=ms,
            block_m=block_m,
        )

    def launch_args(self, topk_ids, topk_weights, n_tokens):
        return (
            ptr_arg(topk_ids),
            ptr_arg(topk_weights),
            ptr_arg(self.sorted_ids),
            ptr_arg(self.sorted_expert_ids),
            ptr_arg(self.num_valid_ids),
            ptr_arg(self.reverse_sorted),
            ptr_arg(self.sorted_weights),
            ptr_arg(self.masked_m),
            ptr_arg(self.m_indices),
            ptr_arg(self.block_offsets),
            ptr_arg(self.real_counts),
            int(n_tokens),
            torch.cuda.current_stream(),
        )


@functools.cache
def compile_moe_sort(
    *,
    E: int,
    topk: int,
    block_m: int,
    sort_ctas: int = 32,
    threads: int = 1024,
    stages: int = 3,
):
    assert sort_ctas % 4 == 0 and (sort_ctas & (sort_ctas - 1)) == 0, (
        "sort_ctas: power of two, multiple of 4"
    )
    assert (block_m & (block_m - 1)) == 0 and threads >= block_m and threads >= E
    experts_per_cta = (E + sort_ctas - 1) // sort_ctas
    bm_shift = int(math.log2(block_m))
    ctas_shift = int(math.log2(sort_ctas))

    allocator = SmemAllocator(
        None, arch=get_rocm_arch(), global_sym_name=f"smem_m3_sort_E{E}"
    )
    lds_count_offset = allocator._align(allocator.ptr, 16)
    lds_padded_offset = lds_count_offset + E * 4
    lds_starts_offset = lds_padded_offset + E * 4
    allocator.ptr = lds_starts_offset + (E + 1) * 4

    tag = f"E{E}_K{topk}_BM{block_m}_C{sort_ctas}_T{threads}"

    def _pairs_and_slice(n_tokens_i32, bx):
        """(total pairs, this CTA's [start, end)) as index values."""
        idx_t = ir.IndexType.get()
        total = arith.index_cast(idx_t, n_tokens_i32 * arith.constant(topk, type=T.i32))
        per_cta = (total + arith.constant(sort_ctas - 1, index=True)) >> arith.constant(
            ctas_shift, index=True
        )
        c_start = bx * per_cta
        c_end = c_start + per_cta
        end = arith.select(arith.cmpi(CmpIPredicate.ult, c_end, total), c_end, total)
        return total, c_start, end

    @flyc.kernel(name=f"m3_sort_count_{tag}", known_block_size=[threads, 1, 1])
    def sort_count(
        arg_topk_ids: fx.Pointer, arg_block_offsets: fx.Pointer, arg_n_tokens: fx.Int32
    ):
        i32 = T.i32
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(i32, tx)
        bx_i32 = arith.index_cast(i32, bx)
        n_tokens_i32 = ArithValue(fx.as_ir_value(arg_n_tokens))

        topk_ptr = extract_global_ptr(arg_topk_ids)
        offsets_ptr = extract_global_ptr(arg_block_offsets)
        base_ptr = allocator.get_base()
        local_count = SmemPtr(base_ptr, lds_count_offset, T.i32, shape=(E,)).get()

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c_threads_idx = arith.constant(threads, index=True)
        c_experts_idx = arith.constant(E, index=True)
        _total, c_start, end = _pairs_and_slice(n_tokens_i32, bx)

        _if_zero = scf.IfOp(arith.cmpi(CmpIPredicate.ult, tx, c_experts_idx))
        with ir.InsertionPoint(_if_zero.then_block):
            memref.store(c0_i32, local_count, [tx])
            scf.YieldOp([])
        gpu.barrier()

        loop = scf.ForOp(c_start + tx, end, c_threads_idx)
        with ir.InsertionPoint(loop.body):
            idx = loop.induction_variable
            eid = global_load_i32(topk_ptr, idx)
            lds_atomic_add_i32(local_count, eid, c1_i32)
            scf.YieldOp([])
        gpu.barrier()

        _if_write = scf.IfOp(arith.cmpi(CmpIPredicate.ult, tx, c_experts_idx))
        with ir.InsertionPoint(_if_write.then_block):
            cnt = memref.load(local_count, [tx])
            off = tx_i32 * arith.constant(sort_ctas, type=i32) + bx_i32
            global_store_i32(offsets_ptr, off, cnt)
            scf.YieldOp([])

    @flyc.kernel(name=f"m3_sort_cumsum_{tag}", known_block_size=[threads, 1, 1])
    def sort_cumsum(
        arg_block_offsets: fx.Pointer,
        arg_masked_m: fx.Pointer,
        arg_real_counts: fx.Pointer,
        arg_cumsum: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
    ):
        i32 = T.i32
        tx = gpu.thread_id("x")
        tx_i32 = arith.index_cast(i32, tx)

        offsets_ptr = extract_global_ptr(arg_block_offsets)
        masked_ptr = extract_global_ptr(arg_masked_m)
        real_ptr = extract_global_ptr(arg_real_counts)
        cumsum_ptr = extract_global_ptr(arg_cumsum)
        expert_ptr = extract_global_ptr(arg_sorted_expert_ids)
        base_ptr = allocator.get_base()
        total_count = SmemPtr(base_ptr, lds_count_offset, T.i32, shape=(E,)).get()
        padded_count = SmemPtr(base_ptr, lds_padded_offset, T.i32, shape=(E,)).get()
        expert_starts = SmemPtr(
            base_ptr, lds_starts_offset, T.i32, shape=(E + 1,)
        ).get()

        c0_i32 = arith.constant(0, type=i32)
        c_sort_ctas = arith.constant(sort_ctas, type=i32)
        c_bm_minus_1 = arith.constant(block_m - 1, type=i32)
        c_bm_mask = arith.constant(~(block_m - 1), type=i32)
        e_valid = arith.cmpi(CmpIPredicate.ult, tx, arith.constant(E, index=True))

        _if_sum = scf.IfOp(e_valid)
        with ir.InsertionPoint(_if_sum.then_block):
            total = c0_i32
            e_base = tx_i32 * c_sort_ctas
            for c_group in range_constexpr(sort_ctas // 4):
                cnts = global_load_i32_vec(
                    offsets_ptr, e_base + arith.constant(c_group * 4, type=i32), 4
                )
                for c_lane in range_constexpr(4):
                    total = total + ArithValue(
                        vector.extract(
                            cnts, dynamic_position=[], static_position=[c_lane]
                        )
                    )
            padded = (total + c_bm_minus_1) & c_bm_mask
            memref.store(total, total_count, [tx])
            memref.store(padded, padded_count, [tx])
            global_store_i32(real_ptr, tx_i32, total)
            global_store_i32(masked_ptr, tx_i32, padded)
            scf.YieldOp([])
        gpu.barrier()

        _if_t0 = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tx_i32, c0_i32))
        with ir.InsertionPoint(_if_t0.then_block):
            scan = scf.ForOp(
                arith.index(0), arith.index(E), arith.index(1), [arith._to_raw(c0_i32)]
            )
            with ir.InsertionPoint(scan.body):
                e = scan.induction_variable
                acc_iter = ArithValue(scan.inner_iter_args[0])
                padded = ArithValue(memref.load(padded_count, [e]))
                memref.store(arith._to_raw(acc_iter), expert_starts, [e])
                scf.YieldOp([arith._to_raw(acc_iter + padded)])
            acc = ArithValue(scan.results[0])
            memref.store(arith._to_raw(acc), expert_starts, [arith.index(E)])
            global_store_i32(cumsum_ptr, c0_i32, acc)
            scf.YieldOp([])
        gpu.barrier()

        _if_update = scf.IfOp(e_valid)
        with ir.InsertionPoint(_if_update.then_block):
            acc = ArithValue(memref.load(expert_starts, [tx]))
            for c in range_constexpr(sort_ctas):
                off = tx_i32 * c_sort_ctas + arith.constant(c, type=i32)
                cnt = global_load_i32(offsets_ptr, off)
                global_store_i32(offsets_ptr, off, acc)
                acc = acc + cnt
            start = ArithValue(memref.load(expert_starts, [tx]))
            end = ArithValue(memref.load(expert_starts, [tx + arith.index(1)]))
            b0 = (start >> arith.constant(bm_shift, type=i32)).index_cast(T.index)
            b1 = (end >> arith.constant(bm_shift, type=i32)).index_cast(T.index)
            fill = scf.ForOp(b0, b1, arith.index(1))
            with ir.InsertionPoint(fill.body):
                b = fill.induction_variable
                global_store_i32(expert_ptr, arith.index_cast(i32, b), tx_i32)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.kernel(name=f"m3_sort_place_pad_{tag}", known_block_size=[threads, 1, 1])
    def sort_place_pad(
        arg_topk_ids: fx.Pointer,
        arg_topk_weight: fx.Pointer,
        arg_block_offsets: fx.Pointer,
        arg_real_counts: fx.Pointer,
        arg_cumsum: fx.Pointer,
        arg_sorted_ids: fx.Pointer,
        arg_reverse_sorted: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_n_tokens: fx.Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(i32, tx)
        bx_i32 = arith.index_cast(i32, bx)
        n_tokens_i32 = ArithValue(fx.as_ir_value(arg_n_tokens))

        topk_ptr = extract_global_ptr(arg_topk_ids)
        topk_weight_ptr = extract_global_ptr(arg_topk_weight)
        offsets_ptr = extract_global_ptr(arg_block_offsets)
        real_ptr = extract_global_ptr(arg_real_counts)
        cumsum_ptr = extract_global_ptr(arg_cumsum)
        sorted_ptr = extract_global_ptr(arg_sorted_ids)
        reverse_ptr = extract_global_ptr(arg_reverse_sorted)
        weights_ptr = extract_global_ptr(arg_sorted_weights)
        mindices_ptr = extract_global_ptr(arg_m_indices)
        base_ptr = allocator.get_base()
        local_offsets = SmemPtr(base_ptr, lds_count_offset, T.i32, shape=(E,)).get()
        row_starts = SmemPtr(base_ptr, lds_starts_offset, T.i32, shape=(E + 1,)).get()

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c_topk = arith.constant(topk, type=i32)
        c_sort_ctas = arith.constant(sort_ctas, type=i32)
        c_threads_idx = arith.constant(threads, index=True)
        c_mask_token = arith.constant(0x00FFFFFF, type=i32)
        c_topk_shift = arith.constant(24, type=i32)
        c_pad = n_tokens_i32 & c_mask_token
        c0_f32 = arith.constant(0.0, type=f32)

        _if_init = scf.IfOp(
            arith.cmpi(CmpIPredicate.ult, tx, arith.constant(E, index=True))
        )
        with ir.InsertionPoint(_if_init.then_block):
            local = global_load_i32(offsets_ptr, tx_i32 * c_sort_ctas + bx_i32)
            start = global_load_i32(offsets_ptr, tx_i32 * c_sort_ctas)
            memref.store(arith._to_raw(local), local_offsets, [tx])
            memref.store(arith._to_raw(start), row_starts, [tx])
            scf.YieldOp([])
        _if_last = scf.IfOp(arith.cmpi(CmpIPredicate.eq, tx_i32, c0_i32))
        with ir.InsertionPoint(_if_last.then_block):
            last = global_load_i32(cumsum_ptr, c0_i32)
            memref.store(arith._to_raw(last), row_starts, [arith.index(E)])
            scf.YieldOp([])
        gpu.barrier()

        _total, c_start, end = _pairs_and_slice(n_tokens_i32, bx)
        place = scf.ForOp(c_start + tx, end, c_threads_idx)
        with ir.InsertionPoint(place.body):
            idx = place.induction_variable
            idx_i32 = arith.index_cast(i32, idx)
            eid = global_load_i32(topk_ptr, idx_i32)
            sp = lds_atomic_add_i32(local_offsets, eid, c1_i32)
            token_id = idx_i32 // c_topk
            slot = idx_i32 % c_topk
            packed = (token_id & c_mask_token) | (slot << c_topk_shift)
            w = global_load_f32(topk_weight_ptr, idx_i32)
            global_store_i32(sorted_ptr, sp, packed)
            global_store_i32(mindices_ptr, sp, token_id & c_mask_token)
            global_store_f32(weights_ptr, sp, w)
            global_store_i32(reverse_ptr, idx_i32, sp)
            scf.YieldOp([])
        gpu.barrier()

        # every expert pads by < block_m rows: lanes [0, block_m) do the padding
        _if_pad = scf.IfOp(
            arith.cmpi(CmpIPredicate.ult, tx, arith.constant(block_m, index=True))
        )
        with ir.InsertionPoint(_if_pad.then_block):
            for ee in range_constexpr(experts_per_cta):
                e = bx * arith.constant(experts_per_cta, index=True) + arith.constant(
                    ee, index=True
                )
                _if_e = scf.IfOp(
                    arith.cmpi(CmpIPredicate.ult, e, arith.constant(E, index=True))
                )
                with ir.InsertionPoint(_if_e.then_block):
                    e_i32 = arith.index_cast(i32, e)
                    start = ArithValue(memref.load(row_starts, [e]))
                    real = global_load_i32(real_ptr, e_i32)
                    padded_end = ArithValue(
                        memref.load(row_starts, [e + arith.index(1)])
                    )
                    j0 = (start + real + tx_i32).index_cast(T.index)
                    j1 = padded_end.index_cast(T.index)
                    pad = scf.ForOp(j0, j1, c_threads_idx)
                    with ir.InsertionPoint(pad.body):
                        j_i32 = arith.index_cast(i32, pad.induction_variable)
                        global_store_i32(sorted_ptr, j_i32, c_pad)
                        global_store_i32(mindices_ptr, j_i32, c_pad)
                        global_store_f32(weights_ptr, j_i32, c0_f32)
                        scf.YieldOp([])
                    scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_sort(
        topk_ids: fx.Pointer,
        topk_weight: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        cumsum: fx.Pointer,
        reverse_sorted: fx.Pointer,
        sorted_weights: fx.Pointer,
        masked_m: fx.Pointer,
        m_indices: fx.Pointer,
        block_offsets: fx.Pointer,
        real_counts: fx.Pointer,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        sort_count(topk_ids, block_offsets, n_tokens).launch(
            grid=(sort_ctas, 1, 1), block=(threads, 1, 1), stream=stream
        )
        if const_expr(stages >= 2):
            sort_cumsum(
                block_offsets, masked_m, real_counts, cumsum, sorted_expert_ids
            ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)
        if const_expr(stages >= 3):
            sort_place_pad(
                topk_ids,
                topk_weight,
                block_offsets,
                real_counts,
                cumsum,
                sorted_ids,
                reverse_sorted,
                sorted_weights,
                m_indices,
                n_tokens,
            ).launch(grid=(sort_ctas, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch_sort


def moe_sort_3stage(
    topk_ids, topk_weights, E, block_m, *, sort_ctas=32, threads=1024, bufs=None
):
    """Drop-in for aiter ``moe_sorting`` (returns ``SortBuffers``)."""
    n_tokens, topk = topk_ids.shape
    if bufs is None:
        bufs = SortBuffers.allocate(
            n_tokens, E, topk, block_m, sort_ctas, topk_ids.device
        )
    launch = compile_moe_sort(
        E=E, topk=topk, block_m=block_m, sort_ctas=sort_ctas, threads=threads
    )
    launch(
        *bufs.launch_args(
            topk_ids.contiguous(), topk_weights.contiguous().float(), n_tokens
        )
    )
    return bufs
