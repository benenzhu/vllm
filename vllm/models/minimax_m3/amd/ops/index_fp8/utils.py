# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers shared by the fp8 indexer kernels: typed global pointers, the compile
cache, the fp8 MFMA, and the wave-level max reduction (permlane swaps + v_max with
nnan)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec


def _global_i32_ptr(addr_i64):
    """Typed global i32 pointer at a raw device address (``p[i]`` loads / stores)."""
    ptr_ty = fx.PointerType.get(
        _T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _bits(f):
    return fx.Float32(f).bitcast(fx.Int32)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _pack8(h0, h1):
    """Two 4 x i32 halves -> the 8 x i32 operand of the fp8 16x16x128 MFMA: VGPRs
    0-3 = K bytes [16*klane, +16), VGPRs 4-7 = [64 + 16*klane, +16) of the lane's
    row (klane = lane // 16)."""
    a, b = Vec(h0), Vec(h1)
    return Vec.from_elements([a[k] for k in range(4)] + [b[k] for k in range(4)], fx.Int32)


def _mfma_fp8_16x16x128(a8, b8, acc4):
    """``v_mfma_f32_16x16x128_f8f6f4`` (fp8 x fp8, whole 128-dim head in one
    instruction, f32 accumulate): the ROCDL scaled op with zero block scales,
    which is what aiter's HIP kernels compile to. An intrinsic rather than inline
    asm, so LLVM still sees the MFMA -> VALU hazard on the accumulator. Lane L
    gets D[4 * (L // 16) + i][L % 16], i = 0..3."""
    zero = fx.as_ir_value(fx.Int32(0))
    e4m3 = _ir.Attribute.parse("#rocdl<matrix_format fp8_e4m3>")
    opsel0 = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(32), 0)
    res = _rocdl_ops.mfma_scale_f32_16x16x128_f8f6f4(
        Vec.make_type(4, fx.Float32),
        fx.as_ir_value(a8),
        fx.as_ir_value(b8),
        fx.as_ir_value(acc4),
        e4m3,  # cbsz: A format
        e4m3,  # blgp: B format
        opsel0,
        zero,  # scale_a (zero block scale)
        opsel0,
        zero,  # scale_b
    ).result
    return Vec(res)


def _permlane16_swap(d_a, d_b):
    """v_permlane16_swap: exchanges the odd 16-lane rows of ``d_a`` with the even
    rows of ``d_b`` (raw rocdl op; fx has no wrapper)."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl.permlane16_swap(
        pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False
    )
    return fx.Int32(_llvm.extractvalue(_T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(_T.i32, res, [1])
    )


def _permlane32_swap(d_a, d_b):
    """v_permlane32_swap: exchanges the upper 32 lanes of ``d_a`` with the lower
    32 lanes of ``d_b``."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = _rocdl.permlane32_swap(
        pair_ty, fx.as_ir_value(d_a), fx.as_ir_value(d_b), False, False
    )
    return fx.Int32(_llvm.extractvalue(_T.i32, res, [0])), fx.Int32(
        _llvm.extractvalue(_T.i32, res, [1])
    )


def _maxf_nn(a, b):
    """v_max_f32 with nnan. Without the flag LLVM canonicalizes both inputs first
    (``v_max x, x, x``, maxnum must quiet sNaNs): 2 extra VALU per max."""
    fm = _ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def _xlane_max4_pair(x, y):
    """4-lane max (lanes L, L^16, L^32, L^48) of two values at once, both broadcast to
    every lane, in 3 permlane swaps (2 x 2 done separately, plus 2 copies):
    swap32(x, y) -> [x_lo|y_lo], [x_hi|y_hi]; max -> m = [X2 | Y2];
    swap16(m, m) + max -> M = [X4 X4 | Y4 Y4]; swap32(M, M) -> [X4|X4], [Y4|Y4]."""
    a, b = _permlane32_swap(_bits(x), _bits(y))
    m = _bits(_maxf_nn(_as_f32(a), _as_f32(b)))
    a, b = _permlane16_swap(m, m)
    big = _bits(_maxf_nn(_as_f32(a), _as_f32(b)))
    a, b = _permlane32_swap(big, big)
    return _as_f32(a), _as_f32(b)
