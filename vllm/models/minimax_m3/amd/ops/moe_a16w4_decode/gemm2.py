# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Decode down GEMM (bf16 intermediate x MXFP4 W2, MFMA 16x16x32) with a
routing-weighted bf16 atomic-add epilogue.

One workgroup per (16-row m-block, TILE_N-column n-block, split-K index); its 4
waves split N. Per TILE_K tile the workgroup copies the 16 x TILE_K bf16 A tile
into LDS (direct-to-LDS loads, XOR-swizzled rows), each wave streams its W2
columns with non-temporal loads (1 KB contiguous per wave instruction in aiter's
preshuffled layout), converts fp4 to bf16 with the per-32 e8m0 scale and runs
the MFMAs. The epilogue stages the f32 accumulators through LDS and adds them,
scaled by the routing weight, into the ``[tokens, hidden]`` output with
packed-bf16 atomics (the output was zeroed by ``sort_decode`` or by gemm1's
pair-0 blocks); the ``ksplit`` CTAs of a tile sum their partials the same way.
The expert id, token ids and routing weights are loaded with the first
instructions so their latency hides under the K loop; padding rows point their
A loads past the buffer's ``num_records`` (zero fill, no traffic) and skip the
atomics. ``pairs``: sort-free routing as in gemm1.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .utils import (
    _a16w4_swizzle_xor16,
    _e8m0_byte_to_f32,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_i32_at,
    _global_i32_buffer_view,
    _lds_ptr3,
    _raw,
    _sconst,
    decode_pairs_table,
    lds_acc_bytes_for,
)
from .utils import (
    buffer_ops as bop,
)

BM = 16


# @flyc.jit is load-bearing: it rewrites ``if token_id < i32_M`` into an scf.if. As a
# plain Python if the guard is dropped at trace time and the atomics fire on padding
# rows (garbage sums, ~13x slower).
@flyc.jit
def _atomic_bf16_epilog(
    lds_acc_base_i32,
    accm,
    arg_out,
    n_block_idx,
    wave,
    lane,
    i32_M,
    N_OUT,
    BN,
    packed,
    weight,
):
    """accm[ni] (f32[4] per lane, MFMA C layout) -> LDS [BM, BN] f32 -> per row: 2
    columns per lane x weight -> packed bf16 atomic add at out[token, col]."""
    _n_per_wave = BN // 4
    num_acc_n = _n_per_wave // 16
    _s_count = BN // 64  # readback: each s-iter covers 64 cols (32 lanes x vec2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_ptr3(lds_acc_base_i32, fx.Int32(0))
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    out_base = _global_base_ptr1(arg_out)

    row_base = lane_div_16 * fx.Int32(4)
    for J in range_constexpr(num_acc_n):
        col = wave * fx.Int32(_n_per_wave) + fx.Int32(J * 16) + lane_mod_16
        vec = Vec(accm[J])
        for v in range_constexpr(4):
            idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
            llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(BM // 8):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(_s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    _gep1(out_base, off),
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


def compile_gemm2(
    *,
    NE,
    N_OUT,
    D_INTER,
    TILE_N=256,
    TILE_K=256,
    ksplit=1,
    b_cache_mod=2,
    waves_per_eu=None,
    pairs=False,
    TOPK=None,
    max_pairs=None,
):
    """N_OUT = hidden size (output columns), D_INTER = contraction. ``ksplit`` CTAs
    per tile each cover D_INTER/ksplit; ``pairs`` needs ``TOPK``."""
    K = D_INTER
    assert K % TILE_K == 0 and K % 256 == 0 and TILE_K % 256 == 0
    assert N_OUT % TILE_N == 0 and (TILE_N // 4) % 16 == 0
    NNB = N_OUT // TILE_N
    KT_ALL = K // TILE_K
    assert KT_ALL % ksplit == 0
    KT = KT_ALL // ksplit  # TILE_K tiles per CTA
    K0 = TILE_K // 128  # 128-K blocks per tile (one 1 KB W block, one scale byte)
    KH_TILE_BYTES = TILE_K * 2  # A bytes per row per tile
    KB16 = KH_TILE_BYTES // 16
    NPW = TILE_N // 4
    NI = NPW // 16
    TILE_K_DW = KH_TILE_BYTES // 4
    NLD = (BM * KH_TILE_BYTES) // (256 * 16)  # A copies per lane per tile
    A_BYTES = BM * KH_TILE_BYTES
    LDS_BYTES = max(
        A_BYTES, lds_acc_bytes_for(BM, TILE_N)
    )  # epilogue reuses the A region
    tab_off = LDS_BYTES  # pairs: 32-entry routing table
    if pairs:
        assert TOPK, "pairs needs TOPK"
        max_pairs = int(max_pairs or BM * TOPK)
        assert max_pairs <= BM * TOPK
        LDS_BYTES += 128
    # W2 preshuffle layout as in gemm1: 16-col x 128-K blocks of 1 KB, lane*16 B inside
    W_BYTES = NE * N_OUT * (K // 2)
    SC_K1 = K // 256
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)
    # padding rows: A loads pointed here (>= num_records 0xFFFFC000) read zeros
    A_OOB_DW = 0x3FFFF000

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    name = (
        f"m3_gemm2_a16w4_ne{NE}_h{N_OUT}_i{K}_tn{TILE_N}_tk{TILE_K}_ks{ksplit}_bcm{b_cache_mod}"
        + (f"_w{waves_per_eu}" if waves_per_eu else "")
        + (f"_pairs{max_pairs}" if pairs else "")
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
        if const_expr(pairs):
            tab = fx.recast_iter(fx.Int32, smem + tab_off)
            e, owner, _, build_tab = decode_pairs_table(
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

            def lds_load_a(ku):
                # K-step ku (8 bf16 per lane): row l16,
                # bytes q16*64 + (ku%4)*16 + (ku//4)*256
                col = q16 * fx.Int32(64) + fx.Int32((ku % 4) * 16 + (ku // 4) * 256)
                byte = l16 * fx.Int32(KH_TILE_BYTES) + _a16w4_swizzle_xor16(
                    l16, col, KB16
                )
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy_atom_call(
                    a_copy_atom, fx.slice(s_x_tiles4, (None, byte // fx.Int32(16))), r
                )
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.BFloat16)

            wr = bop.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_bq)), num_records_bytes=min(W_BYTES, 0xFFFFFFFF)
            )
            sr = bop.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_bscale)), num_records_bytes=min(SW_BYTES, 0xFFFFFFFF)
            )
            # W2 columns of this wave: block (expert_off + col)//16, this CTA's K range
            # starts at 128-K block ks*KT*K0; per tile the block index goes in an SGPR
            col = [
                by_n + wave * fx.Int32(NPW) + fx.Int32(ni * 16)
                for ni in range_constexpr(NI)
            ]
            wvo = [
                lane * fx.Int32(4)
                + ((expert_off + col[ni]) // fx.Int32(16)) * fx.Int32(K // 128 * 256)
                + ks * fx.Int32(KT * K0 * 256)
                for ni in range_constexpr(NI)
            ]
            svo = [
                lane
                + ((expert_off + col[ni]) // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0)
                + ks * fx.Int32(KT * K0 // 2 * 64)
                for ni in range_constexpr(NI)
            ]
            npk = [
                (col[ni] // fx.Int32(16)) % fx.Int32(2) for ni in range_constexpr(NI)
            ]

            def load_w_tile(kt):
                bb = [
                    [
                        bop.buffer_load(
                            wr,
                            wvo[ni] + fx.Int32(((kt * K0 + k0) % 4) * 256),
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=b_cache_mod,
                            soffset_bytes=_sconst(((kt * K0 + k0) // 4) * 4096),
                        )
                        for k0 in range_constexpr(K0)
                    ]
                    for ni in range_constexpr(NI)
                ]
                # one scale dword per 256 K: (kt*K0//2) dwords of 64 -> imm + SGPR
                g2 = kt * K0 // 2
                sc = [
                    bop.buffer_load(
                        sr,
                        svo[ni] + fx.Int32((g2 % 4) * 64),
                        vec_width=1,
                        dtype=fx.Int32,
                        soffset_bytes=_sconst((g2 // 4) * 1024),
                    )
                    for ni in range_constexpr(NI)
                ]
                return bb, sc

            vec2_bf16 = ir.Type.parse("vector<2xbf16>")

            def upconvert(raw4, ku, scale_f32):
                i32_val = _raw(fx.Int32(raw4[ku]))
                s_raw = _raw(scale_f32)
                i32s = []
                for sel in range_constexpr(4):
                    p = rocdl.cvt_scalef32_pk_bf16_fp4(vec2_bf16, i32_val, s_raw, sel)
                    i32s.append(fx.Int32(fx.Vector(p).bitcast(fx.Int32)[0]))
                return fx.Vector.from_elements(
                    [_raw(x) for x in i32s], fx.Int32
                ).bitcast(fx.BFloat16)

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
                        # scale byte: K-half (kt*K0+k0) % 2, N-half npk
                        s = _e8m0_byte_to_f32(
                            fx.Int32(sc[ni]),
                            fx.Int32(((kt * K0 + k0) % 2) * 2) + npk[ni],
                        )
                        raw4 = fx.Vector(bb[ni][k0])
                        for ku in range_constexpr(4):
                            fx.gemm(
                                mma_atom,
                                acc[ni],
                                _frag(lds_load_a(k0 * 4 + ku)),
                                _frag(upconvert(raw4, ku, s)),
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
            **(
                {"value_attrs": {"rocdl.waves_per_eu": waves_per_eu}}
                if waves_per_eu
                else {}
            ),
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
