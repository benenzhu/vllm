# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 prefill MoE final reduce over gemm2's mxfp8 token-major rows
(``out_mode="fp8"``; the a4w4 lab's ``reduce_fp8.py``):

    y[tok, :] = sum_k w[tok, k] * dequant(out[tok*topk + k, :])      -> bf16

One wave per token, 4 tokens per CTA. Lane L owns columns
``i*1024 + L*16 .. +16`` for i in 0..5 (16 fp8 = one dwordx4 per row chunk,
so every load / store instruction of the wave covers 1 KB / 2 KB contiguous).
Per row: 6 dwordx4 + 6 scale bytes, ``v_cvt_pk_f32_fp8`` and one fma per
value with (2^(e8m0-127) * w). Output 16 bf16 per chunk = 2 dwordx4 stores.

Layouts (bytes):
  OUT     [n_tokens*topk, H]      fp8 e4m3, token-major (gemm2)
  OUT_sc  [n_tokens*topk, H/32]   e8m0
  W       [n_tokens, topk]        f32 routing weights (shared expert = 1)
  Y       [n_tokens, H]           bf16
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import vector as _vector
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T as _T
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels import buffer_ops

from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill.gemm1 import _as_f32


def _i1(v: bool):
    return fx.Boolean(v).ir_value()

_TOKENS_PER_CTA = 4


def _cvt_pk_f32_fp8(dword, hi: bool):
    """v_cvt_pk_f32_fp8: 2 fp8 (low / high word of ``dword``) -> 2 f32"""
    v2f32 = _ir.VectorType.get([2], _T.f32)
    res = _llvm.call_intrinsic(v2f32, "llvm.amdgcn.cvt.pk.f32.fp8", [fx.as_ir_value(dword), _i1(hi)], [], [])
    return Vec(res)


def _fma(a, b, c):
    return fx.Float32(
        _llvm.call_intrinsic(_T.f32, "llvm.fma.f32", [fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(c)], [], [])
    )


def _pack_bf16x2(a, b):
    """two f32 -> one i32 of 2 bf16 (RNE, v_cvt_pk_bf16_f32 on gfx950)"""
    v2f32 = _ir.VectorType.get([2], _T.f32)
    v2bf16 = _ir.VectorType.get([2], _ir.BF16Type.get())
    v = _vector.FromElementsOp(v2f32, [fx.as_ir_value(a), fx.as_ir_value(b)]).result
    t = _arith.TruncFOp(v2bf16, v).result
    return _llvm.bitcast(_T.i32, t)


def _v4i32(vals):
    ty = _ir.VectorType.get([4], _T.i32)
    return _vector.FromElementsOp(ty, [fx.as_ir_value(v) for v in vals]).result


def compile_moe_reduce_fp8(*, H: int, topk: int):
    assert H % 1024 == 0
    CHUNKS = H // 1024  # dwordx4 per lane per row
    SC_COLS = H // 32

    @flyc.kernel
    def kernel_reduce(OUT: fx.Tensor, OUT_sc: fx.Tensor, W: fx.Tensor, Y: fx.Tensor, n_tokens: fx.Int32):
        lane = fx.thread_idx.x % 64
        wave = fx.thread_idx.x // 64
        tok = fx.block_idx.x * _TOKENS_PER_CTA + wave
        out_rsrc = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=n_tokens * (topk * H))
        sc_rsrc = buffer_ops.create_buffer_resource(OUT_sc, max_size=False, num_records_bytes=n_tokens * (topk * SC_COLS))
        w_rsrc = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=n_tokens * (topk * 4))
        y_rsrc = buffer_ops.create_buffer_resource(Y, max_size=False, num_records_bytes=n_tokens * (H * 2))
        if tok < n_tokens:
            acc = [[fx.Float32(0.0) for _ in range(16)] for _ in range(CHUNKS)]
            for k in range_constexpr(topk):
                row = tok * topk + k
                w = fx.Float32(buffer_ops.buffer_load(w_rsrc, row, vec_width=1, dtype=fx.Float32))
                row_dw = row * (H // 4)
                for i in range_constexpr(CHUNKS):
                    data = Vec(
                        buffer_ops.buffer_load(out_rsrc, row_dw + i * 256 + lane * 4, vec_width=4, dtype=fx.Int32)
                    )
                    e8 = fx.Int32(buffer_ops.buffer_load(sc_rsrc, row * SC_COLS + i * 32 + lane // 2, vec_width=1, dtype=fx.Int8))
                    e8 = e8 & fx.Int32(0xFF)
                    sw = _as_f32(e8 << 23) * w
                    for d in range_constexpr(4):
                        dw = fx.Int32(data[d])
                        lo = _cvt_pk_f32_fp8(dw, False)
                        hi = _cvt_pk_f32_fp8(dw, True)
                        vals = [fx.Float32(lo[0]), fx.Float32(lo[1]), fx.Float32(hi[0]), fx.Float32(hi[1])]
                        for q in range_constexpr(4):
                            acc[i][4 * d + q] = _fma(vals[q], sw, acc[i][4 * d + q])
            y_dw = tok * (H // 2)  # bf16 row in dwords
            for i in range_constexpr(CHUNKS):
                packed = [_pack_bf16x2(acc[i][2 * j], acc[i][2 * j + 1]) for j in range(8)]
                base = y_dw + i * 512 + lane * 8
                buffer_ops.buffer_store(_v4i32(packed[0:4]), y_rsrc, base)
                buffer_ops.buffer_store(_v4i32(packed[4:8]), y_rsrc, base + 4)

    @flyc.jit
    def launch_reduce(OUT: fx.Tensor, OUT_sc: fx.Tensor, W: fx.Tensor, Y: fx.Tensor, n_tokens: fx.Int32, stream: fx.Stream):
        grid = (n_tokens + _TOKENS_PER_CTA - 1) // _TOKENS_PER_CTA
        kernel_reduce(OUT, OUT_sc, W, Y, n_tokens).launch(grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_reduce
