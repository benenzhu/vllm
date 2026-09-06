# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decode routing sort for the a16w4 MoE chain (M <= max_tokens): one launch.

Block 0 sorts the (token, slot) pairs by expert: every thread loads its <= PPT
pairs at once (one round trip), LDS histogram, a Hillis-Steele scan over the
block_m-padded counts (log2(threads) barrier rounds), then placement through
LDS cursors and padding; every other block zeroes a slice of the output buffer
that gemm2's atomics accumulate into. Same output contract as aiter
``moe_sorting`` / aiter#3832 ``sort_quant_kernel(kSkipQuant)``:

  sorted_ids[row]        = token | slot << 24  (padding: token = n_tokens, slot 0)
  sorted_weights[row]    = routing weight       (padding rows: 0)
  sorted_expert_ids[b]   = expert of BM-row block b
  num_valid_ids[0]       = padded sorted row count
  out[n_tokens, H]       = 0 (bf16)

Rows inside one expert come out in the arrival order of the LDS atomics (not
deterministic, as in aiter); the gemms do not depend on it. aiter's
``moe_sorting`` is two kernels (~9.6 us together at M=64..256 in a graph).
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as _arith_dialect
from flydsl._mlir.dialects import llvm, memref, scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_GEP_DYN = -(2**31)


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _gptr(arg):
    """fx.Pointer argument -> ``!llvm.ptr<1>``."""
    addr_i64 = arith.index_cast(T.i64, fx.ptrtoint(arg))
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(addr_i64))


def _i64(v):
    raw = _raw(v)
    if isinstance(raw.type, ir.IndexType):
        return arith.index_cast(T.i64, raw)
    if ir.IntegerType(raw.type).width == 64:
        return ArithValue(raw)
    return ArithValue(arith.ExtSIOp(T.i64, raw).result)


def _gep(ptr, elem_offset, elem_bytes):
    off = _raw(_i64(elem_offset) * arith.constant(elem_bytes, type=T.i64))
    return llvm.GEPOp(ptr.type, ptr, [off], [_GEP_DYN], T.i8, None).result


def _ld_i32(ptr, off):
    return ArithValue(llvm.LoadOp(T.i32, _gep(ptr, off, 4), alignment=4).result)


def _ld_f32(ptr, off):
    return ArithValue(llvm.LoadOp(T.f32, _gep(ptr, off, 4), alignment=4).result)


def _st_i32(ptr, off, val):
    llvm.StoreOp(_raw(val), _gep(ptr, off, 4), alignment=4)


def _st_vec4_i32(ptr, off16, val):
    llvm.StoreOp(_raw(val), _gep(ptr, off16, 16), alignment=16)


def _lds_atomic_add(lds_memref, elem_offset, value):
    byte_off = ArithValue(_i64(elem_offset) * arith.constant(4, type=T.i64)).index_cast(
        T.index
    )
    base = memref.extract_aligned_pointer_as_index(lds_memref)
    p = llvm.inttoptr(
        ir.Type.parse("!llvm.ptr<3>"), _raw(arith.index_cast(T.i64, base + byte_off))
    )
    return ArithValue(
        llvm.AtomicRMWOp(
            llvm.AtomicBinOp.add,
            p,
            _raw(value),
            llvm.AtomicOrdering.monotonic,
            syncscope="workgroup",
            alignment=4,
        ).result
    )


def _if(cond):
    return scf.IfOp(cond)


def max_sorted_rows(n_tokens: int, E: int, topk: int, block_m: int) -> int:
    """aiter's bound: every active expert pads by < block_m rows."""
    active = min(E, n_tokens * topk)
    cumsum_max = n_tokens * topk + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


@functools.cache
def compile_decode_sort(
    *,
    E: int,
    topk: int,
    block_m: int,
    H: int,
    max_tokens: int = 256,
    zero_ctas: int = 127,
    threads: int = 256,
):
    assert (block_m & (block_m - 1)) == 0 and threads >= E and threads % 64 == 0
    assert (threads & (threads - 1)) == 0, "the scan needs a power-of-two block"
    assert (H * 2) % 16 == 0
    bm_shift = block_m.bit_length() - 1
    PPT = (max_tokens * topk + threads - 1) // threads  # pairs per thread
    scan_rounds = threads.bit_length() - 1

    allocator = SmemAllocator(
        None,
        arch=get_rocm_arch(),
        global_sym_name=f"smem_m3_decode_sort_E{E}_T{threads}",
    )
    off_count = allocator._align(allocator.ptr, 16)
    off_cursor = off_count + E * 4
    off_scan = off_cursor + E * 4  # two ping-pong scan arrays of `threads` entries
    allocator.ptr = off_scan + 2 * threads * 4
    tag = f"E{E}_K{topk}_BM{block_m}_H{H}_M{max_tokens}_Z{zero_ctas}_T{threads}"

    @flyc.kernel(name=f"m3_decode_sort_zero_{tag}", known_block_size=[threads, 1, 1])
    def sort_zero(
        arg_topk_ids: fx.Pointer,
        arg_topk_w: fx.Pointer,
        arg_sorted_ids: fx.Pointer,
        arg_sorted_w: fx.Pointer,
        arg_sorted_eids: fx.Pointer,
        arg_num_valid: fx.Pointer,
        arg_out: fx.Pointer,
        arg_n_tokens: fx.Int32,
    ):
        i32 = T.i32
        idx_t = ir.IndexType.get()
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(i32, tx)
        n_tok = ArithValue(fx.as_ir_value(arg_n_tokens))

        topk_ptr = _gptr(arg_topk_ids)
        w_ptr = _gptr(arg_topk_w)
        sids_ptr = _gptr(arg_sorted_ids)
        sw_ptr = _gptr(arg_sorted_w)
        eids_ptr = _gptr(arg_sorted_eids)
        nvalid_ptr = _gptr(arg_num_valid)
        out_ptr = _gptr(arg_out)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c_threads = arith.constant(threads, index=True)
        c_E = arith.constant(E, index=True)
        c_topk = arith.constant(topk, type=i32)
        c_bm1 = arith.constant(block_m - 1, type=i32)
        c_bm_mask = arith.constant(~(block_m - 1), type=i32)
        c_bm_shift = arith.constant(bm_shift, type=i32)
        c_slot_shift = arith.constant(24, type=i32)
        c_tok_mask = arith.constant(0x00FFFFFF, type=i32)
        c0_f32 = arith.constant(0.0, type=T.f32)
        n_pairs_i32 = n_tok * c_topk
        last_pair = n_pairs_i32 - c1_i32

        is_sorter = arith.cmpi(CmpIPredicate.eq, bx, arith.constant(0, index=True))
        top = scf.IfOp(is_sorter, [], has_else=True)
        with ir.InsertionPoint(top.then_block):
            base_ptr = allocator.get_base()
            count = SmemPtr(base_ptr, off_count, T.i32, shape=(E,)).get()
            cursor = SmemPtr(base_ptr, off_cursor, T.i32, shape=(E,)).get()
            scan_a = SmemPtr(base_ptr, off_scan, T.i32, shape=(threads,)).get()
            scan_b = SmemPtr(
                base_ptr, off_scan + threads * 4, T.i32, shape=(threads,)
            ).get()
            t_lt_E = arith.cmpi(CmpIPredicate.ult, tx, c_E)

            # 0. this thread's pairs: one batch of loads (clamped index, validity mask)
            pairs = []
            for k in range_constexpr(PPT):
                p = tx_i32 + arith.constant(k * threads, type=i32)
                valid = arith.cmpi(CmpIPredicate.slt, p, n_pairs_i32)
                pc = arith.select(valid, p, last_pair)
                pairs.append((p, valid, _ld_i32(topk_ptr, pc), _ld_f32(w_ptr, pc)))

            # 1. zero the counters
            if_z = _if(t_lt_E)
            with ir.InsertionPoint(if_z.then_block):
                memref.store(_raw(c0_i32), count, [tx])
                scf.YieldOp([])
            gpu.barrier()

            # 2. histogram
            for p, valid, e, _w in pairs:
                if_v = _if(valid)
                with ir.InsertionPoint(if_v.then_block):
                    _lds_atomic_add(count, e, c1_i32)
                    scf.YieldOp([])
            gpu.barrier()

            # 3. inclusive Hillis-Steele scan of the padded counts (0 beyond E)
            cnt = ArithValue(
                arith.select(
                    t_lt_E,
                    _raw(
                        memref.load(count, [arith.select(t_lt_E, tx, arith.index(0))])
                    ),
                    _raw(c0_i32),
                )
            )
            padded = (cnt + c_bm1) & c_bm_mask
            memref.store(_raw(padded), scan_a, [tx])
            gpu.barrier()
            src, dst = scan_a, scan_b
            for r in range_constexpr(scan_rounds):
                d = 1 << r
                mine = ArithValue(memref.load(src, [tx]))
                has = arith.cmpi(CmpIPredicate.uge, tx, arith.constant(d, index=True))
                j = arith.select(has, tx - arith.constant(d, index=True), tx)
                other = ArithValue(memref.load(src, [j]))
                v = ArithValue(arith.select(has, _raw(mine + other), _raw(mine)))
                memref.store(_raw(v), dst, [tx])
                gpu.barrier()
                src, dst = dst, src
            incl = ArithValue(memref.load(src, [tx]))
            start = incl - padded
            end = incl

            # 4. thread e: cursor, expert ids of its blocks, padding rows, total
            if_e = _if(t_lt_E)
            with ir.InsertionPoint(if_e.then_block):
                memref.store(_raw(start), cursor, [tx])
                if_last = _if(
                    arith.cmpi(CmpIPredicate.eq, tx, arith.constant(E - 1, index=True))
                )
                with ir.InsertionPoint(if_last.then_block):
                    _st_i32(nvalid_ptr, c0_i32, end)
                    scf.YieldOp([])
                b0 = (start >> c_bm_shift).index_cast(T.index)
                b1 = (end >> c_bm_shift).index_cast(T.index)
                fill = scf.ForOp(b0, b1, arith.index(1))
                with ir.InsertionPoint(fill.body):
                    _st_i32(eids_ptr, fill.induction_variable, tx_i32)
                    scf.YieldOp([])
                r0 = (start + cnt).index_cast(T.index)
                r1 = end.index_cast(T.index)
                pad = scf.ForOp(r0, r1, arith.index(1))
                with ir.InsertionPoint(pad.body):
                    r = pad.induction_variable
                    _st_i32(sids_ptr, r, n_tok & c_tok_mask)
                    _st_i32(sw_ptr, r, c0_f32)
                    scf.YieldOp([])
                scf.YieldOp([])
            gpu.barrier()

            # 5. place the pairs
            for p, valid, e, w in pairs:
                if_v = _if(valid)
                with ir.InsertionPoint(if_v.then_block):
                    row = _lds_atomic_add(cursor, e, c1_i32)
                    packed = ((p // c_topk) & c_tok_mask) | (
                        (p % c_topk) << c_slot_shift
                    )
                    _st_i32(sids_ptr, row, packed)
                    _st_i32(sw_ptr, row, w)
                    scf.YieldOp([])
            scf.YieldOp([])

        with ir.InsertionPoint(top.else_block):
            # blocks 1..zero_ctas: out[n_tokens, H] = 0 as 16-B stores
            i32_ty = ir.IntegerType.get_signless(32)
            v4_ty = ir.VectorType.get([4], i32_ty)
            zero_v = _arith_dialect.ConstantOp(
                v4_ty,
                ir.DenseElementsAttr.get_splat(v4_ty, ir.IntegerAttr.get(i32_ty, 0)),
            ).result
            n_chunks = arith.index_cast(
                idx_t, n_tok * arith.constant(H * 2 // 16, type=i32)
            )
            first = (bx - arith.constant(1, index=True)) * c_threads + tx
            stride = arith.constant(zero_ctas * threads, index=True)
            zl = scf.ForOp(first, n_chunks, stride)
            with ir.InsertionPoint(zl.body):
                _st_vec4_i32(out_ptr, zl.induction_variable, zero_v)
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch(
        topk_ids: fx.Pointer,
        topk_w: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_w: fx.Pointer,
        sorted_eids: fx.Pointer,
        num_valid: fx.Pointer,
        out: fx.Pointer,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        sort_zero(
            topk_ids,
            topk_w,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            out,
            n_tokens,
        ).launch(grid=(1 + zero_ctas, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def _ptr(t: torch.Tensor):
    assert t.is_contiguous()
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _max_tokens_bucket(n_tokens: int) -> int:
    """Compile-time token cap (pairs per thread): 64 / 256 / 1024 / ..."""
    b = 64
    while b < n_tokens:
        b *= 4
    return b


def moe_sort_decode(
    topk_ids,
    topk_weights,
    E,
    H,
    block_m,
    *,
    out=None,
    out_dtype=torch.bfloat16,
    zero_ctas=127,
    threads=256,
):
    """Drop-in for ``moe_sorting(topk_ids, topk_w, E, H, dtype, block_size)`` ->
    (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, zeroed out).
    ``out`` (``[n_tokens, H]`` bf16, contiguous) is zeroed in place when given."""
    n_tokens, topk = topk_ids.shape
    dev = topk_ids.device
    ms = max_sorted_rows(n_tokens, E, topk, block_m)
    sorted_ids = torch.empty(ms, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(ms, dtype=torch.float32, device=dev)
    sorted_eids = torch.empty(ms // block_m, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    if out is None:
        out = torch.empty((n_tokens, H), dtype=out_dtype, device=dev)
    assert (
        out.shape == (n_tokens, H) and out.element_size() == 2 and out.is_contiguous()
    )
    launch = compile_decode_sort(
        E=E,
        topk=topk,
        block_m=block_m,
        H=H,
        max_tokens=_max_tokens_bucket(n_tokens),
        zero_ctas=zero_ctas,
        threads=threads,
    )
    launch(
        _ptr(topk_ids.contiguous().int()),
        _ptr(topk_weights.contiguous().float()),
        _ptr(sorted_ids),
        _ptr(sorted_w),
        _ptr(sorted_eids),
        _ptr(num_valid),
        _ptr(out),
        int(n_tokens),
        torch.cuda.current_stream(),
    )
    return sorted_ids, sorted_w, sorted_eids, num_valid, out
