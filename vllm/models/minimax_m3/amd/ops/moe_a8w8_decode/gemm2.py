# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Decode down GEMM (bf16 intermediate x MXFP8 W2, MFMA 16x16x32) with a
routing-weighted bf16 atomic-add epilogue.

The a16w4 decode gemm2 (``moe_a16w4_decode/gemm2.py``: workgroup per (16-row
m-block, 256-column n-block, split-K index), A tile through LDS with direct
loads, W streamed per wave, LDS-staged atomic epilogue) with fp8 weights:

* a 256-K tile of a column is four 1 KB preshuffled blocks of 64 K (fp4: two of
  128 K); lane (klane, n) holds K ``klane*16 .. +16`` of block ``k0`` -> two MFMA
  K-steps of 8 per block, unpacked with ``v_cvt_scalef32_pk_bf16_fp8``.
* the A fragment of block ``k0``, K-step ``ku`` is row ``l16``, K ``k0*64 +
  q16*16 + ku*8`` of the tile.
* scales: a lane's 16 K lie in 32-K group ``(k0%2)*2 + klane//2`` of the 128-K
  half ``k0//2``, so per 256-K tile it loads two scale dwords (groups
  ``klane//2`` and ``2 + klane//2``, bytes: 128-K half, N-half).

Layouts (aiter ``shuffle_weight(is_guinterleave=True, gate_up=False)`` +
``shuffle_scale``, what vLLM's ``shuffle_mxfp8_moe_weights`` stores for w2):
  W2     [E, H/16, K/64, klane 4, nlane 16, 16 B]  fp8 e4m3
  W2_sc  [E*H/32, K/256, klane 4, nlane 16] dwords  e8m0 (bytes: 128-K half, N-half)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode.gemm2 import _atomic_bf16_epilog
from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode.utils import (
    _a16w4_swizzle_xor16,
    _e8m0_byte_to_f32,
    _gep1,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_view,
    _raw,
    inline_sort_max_pairs,
    inline_sort_table,
)

from .utils import _fp8x8_to_bf16, _pin_sgpr

BM = 16
# Tiles: 128 output columns x 256 K per workgroup (MI355X 09-09 sweep: twice the
# workgroups of the a16w4 256-column tile, 0.5-1 us faster at every M by better
# balance across CUs). Split-K 3 up to KSPLIT_SMALL_M_TOKENS is worth ~1.5 us of
# latency hiding; at 256 tokens it costs ~5 us of extra atomics, so larger batches
# run unsplit.
TILE_N = 128
TILE_K = 256
BLOCK_K = 64  # K per 1 KB W block (16 columns)
KSPLIT_SMALL_M = 3
KSPLIT_SMALL_M_TOKENS = 64
# Kernel arguments loaded up front (_pin_sgpr) for the smallest batches only:
# -0.1..-0.2 us at M <= 4, +0.6 us at M = 16.
PIN_ARGS_TOKENS = 8


def compile_gemm2(
    *,
    NE,
    N_OUT,
    D_INTER,
    n_tokens,
    inline_sort=False,
    TOPK=None,
):
    """N_OUT = hidden size (output columns), D_INTER = contraction. Kernel for
    batches of up to ``n_tokens`` tokens: that picks split-K (``launch.ksplit``
    CTAs per tile, each over D_INTER/ksplit) and the inline-sort scan length
    (``launch.kernel_name``); ``inline_sort`` needs ``TOPK``. ``launch.tile_n`` is
    the N tile for the grid."""
    ksplit = KSPLIT_SMALL_M if n_tokens <= KSPLIT_SMALL_M_TOKENS else 1
    pin_args = inline_sort and n_tokens <= PIN_ARGS_TOKENS
    b_cache_mod = 2  # non-temporal W loads
    K = D_INTER
    assert K % TILE_K == 0 and N_OUT % TILE_N == 0
    NNB = N_OUT // TILE_N
    KT_ALL = K // TILE_K
    KT = KT_ALL // ksplit  # TILE_K tiles per CTA
    K0 = TILE_K // BLOCK_K  # 1 KB W blocks per tile
    KB_ALL = K // BLOCK_K  # W blocks per column group
    KH_TILE_BYTES = TILE_K * 2  # A bytes per row per tile
    KB16 = KH_TILE_BYTES // 16
    NPW = TILE_N // 4
    NI = NPW // 16
    TILE_K_DW = KH_TILE_BYTES // 4
    NLD = (BM * KH_TILE_BYTES) // (256 * 16)  # A copies per lane per tile
    A_BYTES = BM * KH_TILE_BYTES
    LDS_BYTES = max(A_BYTES, BM * TILE_N * 4)  # epilogue reuses the A region
    tab_off = LDS_BYTES  # 32-entry routing table (inline sort only; 128 B)
    LDS_BYTES += 128
    if inline_sort:
        assert TOPK, "inline sort needs TOPK"
        assert n_tokens <= BM, "inline sort: every expert's rows fit one m-block"
        max_pairs = inline_sort_max_pairs(n_tokens, TOPK, BM)
    W_BYTES = NE * N_OUT * K
    SC_K1 = K // 256
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"
    # padding rows: A loads pointed here (>= num_records 0xFFFFC000) read zeros
    A_OOB_DW = 0x3FFFF000

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    name = (
        f"m3_gemm2_a16w8_ne{NE}_h{N_OUT}_i{K}_tn{TILE_N}_tk{TILE_K}_ks{ksplit}_bcm{b_cache_mod}"
        + (f"_isort{max_pairs}" if inline_sort else "")
        + ("_pin" if pin_args else "")
    )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        arg_out: fx.Int64,
    ):
        if const_expr(pin_args):
            # every kernel argument up front (_pin_sgpr)
            arg_a, arg_bq, arg_bscale, arg_stids, arg_sweights, arg_out = (
                _pin_sgpr(a)
                for a in (arg_a, arg_bq, arg_bscale, arg_stids, arg_sweights, arg_out)
            )
            i32_M = _pin_sgpr(i32_M, 32)
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        if const_expr(ksplit > 1):
            tile, ks = pid // fx.Int32(ksplit), pid % fx.Int32(ksplit)
        else:
            tile, ks = pid, fx.Int32(0)
        mb, nb = tile // fx.Int32(NNB), tile % fx.Int32(NNB)
        mrow = mb * fx.Int32(BM)
        m_lane = tx // fx.Int32(32)
        sw_base = _global_base_ptr1(arg_sweights)
        # routing of this block's rows, issued up front: expert id, token id (and
        # weight) per row for the epilogue and the pad mask
        if const_expr(inline_sort):
            tab = fx.recast_iter(fx.Int32, smem + tab_off)
            e, owner, _, build_tab = inline_sort_table(
                arg_stids, i32_M, TOPK, mb, lane, tab, max_pairs=max_pairs
            )
            if owner:
                build_tab()
            np_m1 = i32_M * fx.Int32(TOPK) - fx.Int32(1)
            go = owner

            def stid_at(row):  # token | slot<<24 for row of this block (LDS table)
                return tab[row]

            def sweight_at(row, fused):  # topk_weights[token*TOPK + slot], pads clamped
                f = fx.Int32(fused)
                pair = (f & fx.Int32(0x00FFFFFF)) * fx.Int32(TOPK) + (f >> fx.Int32(24))
                pair = fx.Int32(arith.minsi(_raw(pair), _raw(np_m1)))
                return llvm.load(
                    T.f32, _gep1(sw_base, pair * fx.Int32(4)), invariant=True
                )

        else:
            cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
            e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, mb)))
            stids_base = _global_base_ptr1(arg_stids)
            go = tile < (cumsum0 // fx.Int32(BM)) * fx.Int32(NNB)

            def stid_at(row):
                return llvm.load(
                    T.i32, _gep1(stids_base, (mrow + row) * fx.Int32(4)), invariant=True
                )

            def sweight_at(row, fused):
                return llvm.load(
                    T.f32, _gep1(sw_base, (mrow + row) * fx.Int32(4)), invariant=True
                )

        packed, weight = [], []
        for mr in range_constexpr(BM // 8):
            packed.append(stid_at(fx.Int32(mr * 8) + m_lane))
            weight.append(sweight_at(fx.Int32(mr * 8) + m_lane, packed[-1]))
        # A copies: lane covers dwords tx*4 + i*1024 of the [BM, TILE_K] bf16 tile
        row_local = [
            (tx * fx.Int32(4) + fx.Int32(i * 1024)) // fx.Int32(TILE_K_DW)
            for i in range_constexpr(NLD)
        ]
        col_dw = [
            (tx * fx.Int32(4) + fx.Int32(i * 1024)) % fx.Int32(TILE_K_DW)
            for i in range_constexpr(NLD)
        ]
        row_valid = [
            (fx.Int32(stid_at(row_local[i])) & fx.Int32(0x00FFFFFF)) < i32_M
            for i in range_constexpr(NLD)
        ]

        if go:
            expert_off = e * fx.Int32(N_OUT)
            by_n = nb * fx.Int32(TILE_N)
            xbuf = _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFC000))
            x_tiles4 = fx.logical_divide(xbuf, fx.make_layout(4, 1))
            x_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
            s_x = fx.make_view(
                fx.recast_iter(fx.Int32, smem), fx.make_layout(A_BYTES // 4, 1)
            )
            s_x_tiles4 = fx.logical_divide(s_x, fx.make_layout(4, 1))
            a_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            c_k_dw = (K * 2) // 4
            row_base_dw = [
                (mrow + row_local[i]) * fx.Int32(c_k_dw) for i in range_constexpr(NLD)
            ]

            def dma_a_tile(kt):
                # 16 B per lane straight into LDS; the XOR swizzle is applied to the
                # global column (the LDS destination of a direct load is linear)
                base_dw = (ks * fx.Int32(KT) + fx.Int32(kt)) * fx.Int32(TILE_K_DW)
                for i in range_constexpr(NLD):
                    col_bytes = col_dw[i] * fx.Int32(4)
                    col_sw = _a16w4_swizzle_xor16(row_local[i], col_bytes, KB16)
                    row_k_dw = row_valid[i].select(
                        row_base_dw[i] + base_dw, fx.Int32(A_OOB_DW)
                    )
                    global_byte = row_k_dw * fx.Int32(4) + col_sw
                    lds_byte = row_local[i] * fx.Int32(KH_TILE_BYTES) + col_bytes
                    fx.copy(
                        x_dma_atom,
                        fx.slice(x_tiles4, (None, global_byte // fx.Int32(16))),
                        fx.slice(s_x_tiles4, (None, lds_byte // fx.Int32(16))),
                    )

            def lds_load_a(k0, ku):
                # block k0, K-step ku (8 bf16 per lane): row l16,
                # bytes k0*128 + q16*32 + ku*16
                col = q16 * fx.Int32(32) + fx.Int32(k0 * 128 + ku * 16)
                byte = l16 * fx.Int32(KH_TILE_BYTES) + _a16w4_swizzle_xor16(
                    l16, col, KB16
                )
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy_atom_call(
                    a_copy_atom, fx.slice(s_x_tiles4, (None, byte // fx.Int32(16))), r
                )
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16)

            wr = buffer_ops.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_bq)), num_records_bytes=W_BYTES
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_bscale)), num_records_bytes=SW_BYTES
            )
            # W2 columns of this wave: block (expert_off + col)//16, this CTA's K range
            # starts at 64-K block ks*KT*K0; per tile the block index goes in an SGPR
            col = [
                by_n + wave * fx.Int32(NPW) + fx.Int32(ni * 16)
                for ni in range_constexpr(NI)
            ]
            wvo = [
                lane * fx.Int32(4)
                + ((expert_off + col[ni]) // fx.Int32(16)) * fx.Int32(KB_ALL * 256)
                + ks * fx.Int32(KT * K0 * 256)
                for ni in range_constexpr(NI)
            ]
            # scale dwords of 32-K groups klane//2 (b = 0) and 2 + klane//2 (b = 1)
            lane_sc = [
                (fx.Int32(2 * b) + q16 // fx.Int32(2)) * fx.Int32(16) + l16
                for b in range_constexpr(2)
            ]
            svo = [
                [
                    lane_sc[b]
                    + ((expert_off + col[ni]) // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0)
                    + ks * fx.Int32(KT * 64)
                    for ni in range_constexpr(NI)
                ]
                for b in range_constexpr(2)
            ]
            npk = [
                (col[ni] // fx.Int32(16)) % fx.Int32(2) for ni in range_constexpr(NI)
            ]

            def load_w_tile(kt):
                bb = [
                    [
                        buffer_ops.buffer_load(
                            wr,
                            wvo[ni] + fx.Int32(((kt * K0 + k0) % 4) * 256),
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=b_cache_mod,
                            soffset_bytes=((kt * K0 + k0) // 4) * 4096,
                        )
                        for k0 in range_constexpr(K0)
                    ]
                    for ni in range_constexpr(NI)
                ]
                # two scale dwords per 256-K tile: (kt % 4) dwords of 64 -> imm + SGPR
                sc = [
                    [
                        buffer_ops.buffer_load(
                            sr,
                            svo[b][ni] + fx.Int32((kt % 4) * 64),
                            vec_width=1,
                            dtype=fx.Int32,
                            soffset_bytes=(kt // 4) * 1024,
                        )
                        for ni in range_constexpr(NI)
                    ]
                    for b in range_constexpr(2)
                ]
                return bb, sc

            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            acc_layout = fx.make_layout(4, 1)
            acc = [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(NI)]
            zero4 = Vec.filled(4, 0.0, fx.Float32)
            for ni in range_constexpr(NI):
                acc[ni].store(zero4)

            def _frag(v8):
                t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                t.store(v8)
                return t

            for kt in range_constexpr(KT):
                dma_a_tile(kt)
                bb, sc = load_w_tile(kt)
                gpu.barrier()
                for ni in range_constexpr(NI):
                    for k0 in range_constexpr(K0):
                        # scale byte: 128-K half k0//2 -> +2, N-half npk; dword k0%2
                        s = _e8m0_byte_to_f32(
                            fx.Int32(sc[k0 % 2][ni]),
                            fx.Int32((k0 // 2) * 2) + npk[ni],
                        )
                        raw4 = fx.Vector(bb[ni][k0])
                        for ku in range_constexpr(2):
                            fx.gemm(
                                mma_atom,
                                acc[ni],
                                _frag(lds_load_a(k0, ku)),
                                _frag(_fp8x8_to_bf16(raw4[2 * ku], raw4[2 * ku + 1], s)),
                                acc[ni],
                            )
                gpu.barrier()

            # epilogue: the A region is free once every wave passed the last barrier
            gpu.barrier()
            _atomic_bf16_epilog(
                fx.Int32(fx.ptrtoint(smem)),
                [acc[ni].load().ir_value() for ni in range(NI)],
                arg_out,
                nb,
                wave,
                lane,
                i32_M,
                N_OUT,
                TILE_N,
                packed,
                weight,
            )

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_grid: fx.Int32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        kernel(
            arg_a,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            arg_out,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.tile_n = TILE_N
    launch.ksplit = ksplit
    return launch
