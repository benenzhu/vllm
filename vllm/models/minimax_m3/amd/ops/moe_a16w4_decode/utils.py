# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Shared low-level helpers for the a16w4/a16wi4/a16w16 fused MoE kernels
(:mod:`gemm1` stage1 and :mod:`gemm2` stage2). Pointer/GEP builders, buffer-tensor
views, e8m0/int4 dequant, the A-LDS XOR swizzle, and the arch gate."""

import os
import re

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

try:  # flydsl >= 0.3 moved the raw buffer ops out of flydsl.expr; aiter ships a copy
    from aiter.ops.flydsl.kernels import buffer_ops
except ImportError:  # flydsl <= 0.2.x
    from flydsl.expr import buffer_ops

_PTR3 = "!llvm.ptr<3>"
LOG2E = 1.4426950408889634

# a16wi4 (int4 W) groupwise scale: group_size = 32 == one MFMA K32 step (one ku per
# K-group). Scale packed bf16 pairs (E, N, G//2, 2); even/odd ku selects lo/hi half.
A16WI4_GROUP_SIZE = 32


def a16wmix_use_k16(arch=None):
    """True for the gfx942 (CDNA3) codepath: K=16 MFMA + scalar int4 dequant.

    Arch-gate: gfx950 (CDNA4) has K=32 mfma_f32_16x16x32_bf16 + v_cvt_pk_bf16_f32;
    gfx942 has neither and falls back to K=16 MFMA + scalar-trunc dequant.
    ``FLYDSL_A16WMIX_FORCE_K16=1`` forces the gfx942 path (a strict ISA subset) for
    validation on a gfx950 box.
    """
    if os.environ.get("FLYDSL_A16WMIX_FORCE_K16", "0") not in (
        "0",
        "",
        "false",
        "False",
    ):
        return True
    if arch is None:
        arch = get_rocm_arch() or ""
    return "gfx95" not in str(arch)


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def s_waitcnt_lgkm0():
    """``s_waitcnt lgkmcnt(0)`` (vmcnt/expcnt left at max). flydsl 0.2.x exposes only
    the
    raw simm16 form; CDNA encoding: vmcnt [3:0]+[15:14], expcnt [6:4], lgkmcnt
    [11:8]."""
    return rocdl.s_waitcnt(0xC07F)


def _parse_static_strides(layout):
    ly_str = str(layout.type) if hasattr(layout, "type") else str(layout)
    m = re.search(r"\(([^)]+)\):\(([^)]+)\)", ly_str)
    assert m, f"cannot parse layout {ly_str!r}"
    strides = [t.strip() for t in m.group(2).split(",")]
    assert all(t != "?" for t in strides), (
        f"crd2idx needs static strides, got {ly_str!r}"
    )
    return [int(t) for t in strides]


def crd2idx(crd, layout):
    """Flat index of a coordinate list under a STATIC-stride layout: sum(c_i *
    stride_i).
    (Trimmed copy of kernels/common/layout_utils.crd2idx; our W/scale layouts are
    static.)"""
    strides = _parse_static_strides(layout)
    assert len(crd) == len(strides), (len(crd), strides)
    acc = None
    for c, st in zip(crd, strides):
        if st == 0:
            continue
        term = c * fx.Int64(st) if st != 1 else c
        acc = term if acc is None else acc + term
    return acc


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


def _global_i32_buffer_view(addr_i64, num_bytes):
    # fx.copy BufferCopy atoms take soffset as an element count (not bytes); the
    # make_layout dynamic-shape leaf must be i32/i64, not fx.Index.
    num_bytes_i64 = fx.Int64(num_bytes)
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1)))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=False, num_records_bytes=num_bytes_i64
    )


def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
    return fx.logical_divide(
        _global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1)
    )


def _buffer_i32_scalar_read(tiles1, idx, atom):
    """Read one i32 dword at element ``idx`` from a ``_global_i32_buffer_tiles(..., 1)``
    view via the layout-API BufferCopy atom (buffer_load_dword; OOB-clamped by the
    buffer resource). ``tiles1`` is 1-dword tiles so the tile index == ``idx``.
    """
    r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    fx.copy(atom, fx.slice(tiles1, (None, idx)), r)
    return fx.Int32(fx.Vector(fx.memref_load_vec(r))[0])


def decode_pairs_table(arg_topk, i32_ntok, TOPK, p_i32, lane, tab_ptr3, max_pairs=64):
    """Sort-free decode routing (n_tokens <= BM): build this block's sorted_token_ids
    table.

    Routing pair q = token*TOPK + slot (row-major topk_ids). Block p owns expert
    e = topk_ids[p] iff p is the FIRST pair with that expert; its rows are all pairs
    with
    expert e, in pair order (<= n_tokens <= BM rows, so one m-block per expert; the
    shared
    expert is the block with n_tokens rows). The 32-entry table at ``tab_ptr3`` (LDS)
    holds token | slot<<24 per row, token = n_tokens for padding rows, i.e. exactly what
    moe_sorting would have written for this block. Returns (expert id, owner); non-owner
    blocks (duplicate experts) must exit. Pairs are scanned 64 per wave pass;
    ``max_pairs``
    (BM*TOPK, 80 at BM=16 / topk 5) sets the number of passes. One 80-320 B load +
    ballots + LDS stores per block instead of a separate sort kernel.
    """
    n_chunks = (int(max_pairs) + 63) // 64
    n_pairs = i32_ntok * fx.Int32(TOPK)
    last = n_pairs - fx.Int32(1)
    p_lane = p_i32 % fx.Int32(64)
    p_chunk = p_i32 // fx.Int32(64)
    qs, pvs = [], []
    for c in range_constexpr(n_chunks):
        q = lane + fx.Int32(c * 64)
        idx = fx.Int32(arith.minsi(_raw(q), _raw(last)))
        v = fx.Int32(_global_i32_at(arg_topk, idx))
        qs.append(q)
        pvs.append((q < n_pairs).select(v, fx.Int32(-1)))
    # my expert = value of pair p (uniform)
    e = fx.Int32(rocdl.readlane(T.i32, _raw(pvs[0]), _raw(p_lane)))
    for c in range_constexpr(1, n_chunks):
        e_c = fx.Int32(rocdl.readlane(T.i32, _raw(pvs[c]), _raw(p_lane)))
        e = (p_chunk == fx.Int32(c)).select(e_c, e)
    base = fx.Int32(0)
    rank_p = fx.Int32(0)
    slots, fuseds = [], []
    for c in range_constexpr(n_chunks):
        is_match = pvs[c] == e
        mask = rocdl.ballot(T.i64, _raw(is_match))
        mask_lo = arith.trunci(T.i32, mask)
        mask_hi = arith.trunci(T.i32, arith.shrui(mask, arith.constant(32, type=T.i64)))
        below = fx.Int32(
            rocdl.mbcnt_hi(
                T.i32, mask_hi, rocdl.mbcnt_lo(T.i32, mask_lo, _raw(fx.Int32(0)))
            )
        )
        rank = (
            base + below
        )  # row of pair q among this expert's pairs (global pair order)
        fuseds.append(
            (qs[c] // fx.Int32(TOPK)) | ((qs[c] % fx.Int32(TOPK)) << fx.Int32(24))
        )
        slots.append(
            is_match.select(rank, fx.Int32(31))
        )  # non-matching lanes park in slot 31
        r_p = fx.Int32(rocdl.readlane(T.i32, _raw(rank), _raw(p_lane)))
        rank_p = (p_chunk == fx.Int32(c)).select(r_p, rank_p)
        # matches so far = rank at lane 63 + lane 63's own match bit (row count after
        # the last chunk)
        tot = rank + is_match.select(fx.Int32(1), fx.Int32(0))
        base = fx.Int32(rocdl.readlane(T.i32, _raw(tot), _raw(fx.Int32(63))))
    owner = rank_p == fx.Int32(0)

    def build_table():
        # padding sentinel in every slot first (slots 0..31), then the rows
        llvm.StoreOp(
            _raw(i32_ntok), _gep3(tab_ptr3, (lane % fx.Int32(32)) * fx.Int32(4))
        )
        gpu.barrier()
        for c in range_constexpr(n_chunks):
            llvm.StoreOp(_raw(fuseds[c]), _gep3(tab_ptr3, slots[c] * fx.Int32(4)))
        gpu.barrier()

    # The kernel calls build_table() under `if owner:` (a uniform branch the kernel's
    # AST
    # rewriter turns into scf.if), so duplicate-expert blocks leave after the loads and
    # ballots without the two barriers (36% of the blocks at M=16).
    return e, owner, base, build_table


def wave_count(pred, lane):
    """Number of lanes (wave64) where the Boolean ``pred`` holds, as a uniform Int32."""
    mask = rocdl.ballot(T.i64, _raw(pred))
    mask_lo = arith.trunci(T.i32, mask)
    mask_hi = arith.trunci(T.i32, arith.shrui(mask, arith.constant(32, type=T.i64)))
    below = fx.Int32(
        rocdl.mbcnt_hi(
            T.i32, mask_hi, rocdl.mbcnt_lo(T.i32, mask_lo, _raw(fx.Int32(0)))
        )
    )
    tot = below + pred.select(fx.Int32(1), fx.Int32(0))
    return fx.Int32(rocdl.readlane(T.i32, _raw(tot), _raw(fx.Int32(63))))


def _lds_ptr3(base_i32, byte_off_i32):
    addr_i64 = fx.Int64(base_i32 + byte_off_i32)
    return llvm.inttoptr(ir.Type.parse(_PTR3), _raw(addr_i64))


def _gep3(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_base_ptr1(addr_i64):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def _gep1(base_ptr, byte_off_i32):
    return buffer_ops.get_element_ptr(
        base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8
    )


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    return _global_i32_ptr(addr_i64)[idx]


def _e8m0_byte_to_f32(packed_i32, byte_pos):
    shift = byte_pos * fx.Int32(8)
    b = packed_i32.shrui(shift) & fx.Int32(0xFF)
    return fx.Float32(_raw(b << fx.Int32(23)).bitcast(T.f32))


def _cvt_pk_bf16_f32_se(src_a_f32, src_b_f32):
    # Side-effecting v_cvt_pk_bf16_f32 (pack 2 f32 -> 2xbf16 in i32). LOAD-BEARING:
    # the stateless rocdl.cvt_pk_bf16_f32 gets CSE-merged/reordered across K steps in
    # the a16wi4 gemm1 hot loop (garbage output); side_effects pins each call.
    return llvm.inline_asm(
        ir.IntegerType.get_signless(32),
        [_raw(src_a_f32), _raw(src_b_f32)],
        "v_cvt_pk_bf16_f32 $0, $1, $2",
        "=v,v,v",
        has_side_effects=True,
    )


def _int4_nibble_to_bf16x8(raw_i32, scale_f32, *, use_k16=False):
    """int4 (signed) -> bf16 upconvert for one MFMA K32 step (8 nibbles -> v8bf16).

    ``raw_i32`` holds 8 signed-int4 nibbles in bits[4n+3:4n] (same K order as the
    mxfp4 sel 0..3 path). ``v_cvt_off_f32_i4`` reads the nibble unsigned, subtracts 8,
    and scales the mantissa by 16, so the x16 is folded into eff = scale*16.
    ``use_k16`` (gfx942): v_cvt_pk_bf16_f32 is gfx950-only -> scalar .to(BFloat16).
    """
    eff = fx.Float32(scale_f32 * fx.Float32(16.0))
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    if use_k16:
        # gfx942 fallback: scalar f32 -> bf16 truncation (no v_cvt_pk_bf16_f32).
        bf16s = []
        for j in range_constexpr(4):
            f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
            f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
            bf16s.append(f_lo.to(fx.BFloat16))
            bf16s.append(f_hi.to(fx.BFloat16))
        return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
    # byte_sel loads (1 shift total); side-effecting pk-convert.
    i32s = []
    for j in range_constexpr(4):
        f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j)) * eff
        f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j)) * eff
        i32s.append(fx.Int32(_cvt_pk_bf16_f32_se(_raw(f_lo), _raw(f_hi))))
    v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
    return v4i32.bitcast(fx.BFloat16)  # v8bf16


def _int4_nibble_to_bf16x8_raw(raw_i32, *, use_k16=False):
    """int4 (signed) -> bf16 for one MFMA K32 step WITHOUT the groupwise scale.

    Same as :func:`_int4_nibble_to_bf16x8` but emits the raw dequant weights
    ``(nibble-8)/16`` (``v_cvt_off_f32_i4``'s native output -- no per-element
    ``v_mul_f32``). The groupwise scale (and the folded x16) is applied ONCE per
    K-group on the small MFMA accumulator instead (see the ``_acc_scale_int4`` path in
    the stage1 body): for BM16 (m_repeat=1) that trades 8 per-nibble muls for 4
    per-accumulator fmas and drops the long-lived scaled-f32 operand VGPRs.
    ``(nibble-8)/16`` is bf16-exact (values in ``{-7/16..7/16}``).
    ``use_k16`` (gfx942): v_cvt_pk_bf16_f32 is gfx950-only -> scalar .to(BFloat16).
    """
    raw_even = fx.Int32(raw_i32)
    raw_odd = raw_even.shrui(fx.Int32(4))
    if use_k16:
        bf16s = []
        for j in range_constexpr(4):
            f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j))
            f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j))
            bf16s.append(f_lo.to(fx.BFloat16))
            bf16s.append(f_hi.to(fx.BFloat16))
        return fx.Vector.from_elements([_raw(x) for x in bf16s], fx.BFloat16)  # v8bf16
    i32s = []
    for j in range_constexpr(4):
        f_lo = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_even), byte_sel=j))
        f_hi = fx.Float32(rocdl.cvt_off_f32_i4(_raw(raw_odd), byte_sel=j))
        i32s.append(fx.Int32(_cvt_pk_bf16_f32_se(_raw(f_lo), _raw(f_hi))))
    v4i32 = fx.Vector.from_elements([_raw(x) for x in i32s], fx.Int32)
    return v4i32.bitcast(fx.BFloat16)  # v8bf16


def kmchunks_for(BM):
    return BM // 16


def lds_acc_bytes_for(rows, BN):
    return rows * BN * 4


def _a16w4_swizzle_xor16(row, col_bytes, k_blocks16, *, enable=False):
    """A-LDS bank-conflict XOR swizzle (aiter swizzle_xor16: col ^ ((row&(kb16-1))*16)).

    Both the DMA write and the LDS read go through this helper so the physical layout
    stays consistent. gemm1 keeps linear (enable=False); gemm2 enables it.
    """
    if not enable:
        return col_bytes
    rem = row & fx.Int32(k_blocks16 - 1)
    return col_bytes ^ (rem * fx.Int32(16))
