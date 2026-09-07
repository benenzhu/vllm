# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode gate/up GEMM (bf16 x MXFP4 W13, MFMA 16x16x32) fused with swiglu-OAI.

One workgroup per (16-row m-block, TILE_N-column n-block). Its 4 waves split N
(``k_waves`` 1: four N-waves of TILE_N/4 columns; 2: two N-waves x two K-waves
with an LDS reduce at the end) and share the A rows: the workgroup loads one
batch of ``k_batch`` 128-K tiles of the 16 rows (one dwordx4 per lane per
K-tile, each instruction 4 rows x 256 B = 8 full cache lines) into LDS and every
wave reads its MFMA A fragments from there. Two LDS slots, one barrier per batch;
the batch loads go out before the same iteration's W loads so the in-order vmcnt
wait for that W tile also covers them. W streams through a VGPR ring
``prefetch`` tiles deep with non-temporal loads (1 KB contiguous per wave
instruction in aiter's preshuffled layout), the per-32 e8m0 scales come one
packed dword per 256 K, and the fp4 -> bf16 conversion runs right before each
MFMA.

Two things the compiler needs here: a scheduling barrier after every K-tile
(otherwise it hoists the conversions of the next tiles above the current MFMAs
and spills), and the K position of the loads in an opaque SGPR (``_sconst``;
otherwise every uniform offset is folded into a per-tile vector address).

Routing: ``pairs`` (n_tokens <= 16) derives the expert and rows of each block
from ``topk_ids`` with a wave ballot (``utils.decode_pairs_table``) and the
blocks of pair 0 zero the stage-2 output; otherwise the rows come expert-sorted
from ``sort_decode``. Padding rows carry a token id >= n_tokens: their A loads
read 0 through the OOB-clamped buffer resource and their outputs are masked.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .utils import (
    _e8m0_byte_to_f32,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_i32_at,
    _lds_ptr3,
    _raw,
    _sconst,
    _swigluoai_f32,
    decode_pairs_table,
    s_waitcnt_lgkm0,
)
from .utils import (
    buffer_ops as bop,
)

BM = 16  # rows per m-block (one MFMA M tile)
NW = 4  # waves per workgroup
LDS_PAD = 16  # bytes of padding per LDS row: conflict-free 16 B reads


def compile_gemm1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    TILE_N=32,
    k_waves=2,
    k_batch=2,
    prefetch=3,
    b_cache_mod=2,
    w_layout="standard",
    waves_per_eu=None,
    pairs=False,
    max_pairs=None,
):
    K, INTER = D_HIDDEN, D_INTER
    N_OUT = 2 * INTER
    KW = k_waves
    assert KW in (1, 2) and TILE_N in (32, 64, 128, 256)
    assert w_layout in ("standard", "guinterleave")
    NWN = NW // KW  # N-waves
    NPW = TILE_N // NWN  # columns per wave (gate and up each)
    NI = NPW // 16
    assert NPW % 16 == 0 and INTER % TILE_N == 0 and K % 256 == 0
    KT = K // 128  # 128-K tiles: 1 KB of W per 16 columns, 256 B of A per row
    KTW = KT // KW  # K-tiles per K-wave
    KB = k_batch
    assert KT % KW == 0 and KTW % KB == 0 and KB % 2 == 0 and KTW % 2 == 0
    assert 1 <= prefetch < KTW
    NNB = INTER // TILE_N
    ROWB = KB * 256  # A bytes per row per batch (per K-wave)
    RS = ROWB + LDS_PAD  # LDS row stride
    KSLOT = BM * RS  # one K-wave's batch
    SLOT = KW * KSLOT
    RED_BYTES = (KW - 1) * NWN * 2 * NI * 1024  # K-reduce scratch (reuses the A slots)
    LDS_BYTES = max(2 * SLOT, RED_BYTES)
    tab_off = LDS_BYTES  # pairs: 32-entry routing table
    if pairs:
        max_pairs = int(max_pairs or BM * TOPK)
        assert max_pairs <= BM * TOPK
        LDS_BYTES += 128
    # W (mxfp4) preshuffle layout (aiter make_preshuffle_b_layout, N-major, fp4 bytes):
    # (N_OUT/16, K/128, klane 4, nlane 16, kpack 16 B): one 16-col x 128-K block is 1 KB
    # contiguous; lane (klane, n) holds the 32 K of column n at klane -> lane*16 B.
    W_BYTES = NE * N_OUT * (K // 2)
    # scale preshuffle (make_preshuffle_scale_layout, e8m0 u8): (N_OUT/32, K/256,
    # klane 4, nlane 16) dwords; one dword = 2 K-tiles x 2 N-halves (16 cols each).
    SC_K1 = K // 256
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    name = (
        f"m3_gemm1_a16w4_h{K}_i{INTER}_ne{NE}_tn{TILE_N}_kw{KW}_kb{KB}_pf{prefetch}"
        f"_bcm{b_cache_mod}"
        + ("" if w_layout == "standard" else "_gu")
        + (f"_w{waves_per_eu}" if waves_per_eu else "")
        + (f"_pairs{max_pairs}" if pairs else "")
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_x: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        f32_alpha: fx.Float32,
        f32_limit: fx.Float32,
        arg_out: fx.Int64,
        arg_zero: fx.Int64,
        i32_zero_dw: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        pid = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tx // fx.Int32(64))))
        l16, q16 = lane % fx.Int32(16), lane // fx.Int32(16)
        wave_n, wave_k = wave % fx.Int32(NWN), wave // fx.Int32(NWN)
        mb, nb = pid // fx.Int32(NNB), pid % fx.Int32(NNB)
        mbase = mb * fx.Int32(BM)
        if const_expr(pairs):
            # block = routing pair mb: expert + rows from a ballot over the pairs
            tab = _lds_ptr3(fx.Int32(fx.ptrtoint(smem)), fx.Int32(tab_off))
            e_pair, owner, _nrows, build_tab = decode_pairs_table(
                arg_mind, i32_ntok, TOPK, mb, lane, tab, max_pairs=max_pairs
            )
            if owner:
                build_tab()
            cumsum0 = i32_ntok * fx.Int32(TOPK * BM)
            go = owner
            # zero the stage-2 output (gemm2 accumulates with atomics): the NNB blocks
            # of pair 0 stride over it, one dword per thread
            if mb == fx.Int32(0):
                zb = _global_base_ptr1(arg_zero)
                for iv in range(
                    pid * fx.Int32(64 * NW) + tx, i32_zero_dw, NNB * 64 * NW
                ):
                    llvm.StoreOp(
                        _raw(fx.Int32(0)), _gep1(zb, fx.Int32(iv) * fx.Int32(4))
                    )

            def mind_at(row):
                return fx.Int32(llvm.load(T.i32, _gep3(tab, row * fx.Int32(4))))

        else:
            cumsum0 = fx.Int32(_global_i32_at(arg_cumsum, fx.Int32(0)))
            go = mbase < cumsum0

            def mind_at(row):
                return fx.Int32(_global_i32_at(arg_mind, mbase + row))

        if go:
            if const_expr(pairs):
                e = e_pair
            else:
                e = fx.Int32(
                    rocdl.readfirstlane(
                        T.i32, _raw(fx.Int32(_global_i32_at(arg_eids, mb)))
                    )
                )
            # A staging: wave w loads row w*4 + lane//16, 16 B chunk j*16 + lane%16
            ld_row = wave * fx.Int32(4) + q16
            ld_tok = mind_at(ld_row) & fx.Int32(0xFFFFFF)
            # epilogue rows: lane (q16, l16) holds rows q16*4 + ii of column l16
            ep_tok = [
                mind_at(q16 * fx.Int32(4) + fx.Int32(ii)) & fx.Int32(0xFFFFFF)
                for ii in range_constexpr(4)
            ]
            xr = bop.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_x)),
                num_records_bytes=_raw(fx.Int64(i32_ntok) * fx.Int64(K * 2)),
            )
            wr = bop.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_bq)), num_records_bytes=min(W_BYTES, 0xFFFFFFFF)
            )
            sr = bop.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_bscale)), num_records_bytes=min(SW_BYTES, 0xFFFFFFFF)
            )
            outr = bop.create_buffer_resource_from_addr(
                _raw(fx.Int64(arg_out)),
                num_records_bytes=_raw(fx.Int64(cumsum0) * fx.Int64(INTER * 2)),
            )
            lds = llvm.inttoptr(
                ir.Type.parse("!llvm.ptr<3>"),
                fx.as_ir_value(fx.Int32(fx.ptrtoint(smem))),
            )
            ld_gdw = (ld_tok * fx.Int32(K * 2) + l16 * fx.Int32(16)) // fx.Int32(4)
            ld_lbyte = ld_row * fx.Int32(RS) + l16 * fx.Int32(16)

            def load_a_batch(b):
                # batch b of every K-wave: tiles kw*KTW + b*KB + j; base in an SGPR,
                # j*256 in the immediate offset field
                out = []
                for kw in range_constexpr(KW):
                    so = _sconst((kw * KTW + b * KB) * 256)
                    out += [
                        bop.buffer_load(
                            xr,
                            ld_gdw + fx.Int32(j * 64),
                            vec_width=4,
                            dtype=fx.Int32,
                            soffset_bytes=so,
                        )
                        for j in range_constexpr(KB)
                    ]
                return out

            def stage_a_batch(regs, slot):
                for kw in range_constexpr(KW):
                    for j in range_constexpr(KB):
                        ptr = bop.get_element_ptr(
                            lds,
                            byte_offset=fx.as_ir_value(
                                ld_lbyte + fx.Int32(slot * SLOT + kw * KSLOT + j * 256)
                            ),
                            elem_type=T.i8,
                        )
                        llvm.StoreOp(
                            fx.as_ir_value(regs[kw * KB + j]), ptr, alignment=16
                        )
                s_waitcnt_lgkm0()
                gpu.barrier()

            # MFMA A fragment (K-step ku of tile kt): row l16, K = klane*32 + ku*8
            rd_base = wave_k * fx.Int32(KSLOT) + l16 * fx.Int32(RS) + q16 * fx.Int32(64)

            def read_a_tile(kt):
                slot, j = (kt // KB) % 2, kt % KB
                out = []
                for ku in range_constexpr(4):
                    ptr = bop.get_element_ptr(
                        lds,
                        byte_offset=fx.as_ir_value(
                            rd_base + fx.Int32(slot * SLOT + j * 256 + ku * 16)
                        ),
                        elem_type=T.i8,
                    )
                    out.append(
                        fx.Vector(
                            llvm.load(T.vec(4, T.i32), ptr, alignment=16)
                        ).bitcast(fx.BFloat16)
                    )
                return out

            # W addressing (gu 0 = gate, 1 = up), NI 16-column tiles per wave
            nbase = nb * fx.Int32(TILE_N) + wave_n * fx.Int32(NPW)
            if const_expr(w_layout == "guinterleave"):
                # gate/up 16-row blocks interleaved; the scale dword packs gate
                # (bytes 0/2) and up (1/3)
                n0 = [
                    (nbase + fx.Int32(ni * 16)) // fx.Int32(16)
                    for ni in range_constexpr(NI)
                ]
                nblk = [
                    [
                        e * fx.Int32(N_OUT // 16) + n0[ni] * fx.Int32(2) + fx.Int32(gu)
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                mni = [
                    [e * fx.Int32(N_OUT // 32) + n0[ni] for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
                npk = [
                    [fx.Int32(gu) for _ in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
            else:
                ng = [
                    [
                        e * fx.Int32(N_OUT) + nbase + fx.Int32(ni * 16 + gu * INTER)
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                nblk = [
                    [ng[gu][ni] // fx.Int32(16) for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
                mni = [
                    [ng[gu][ni] // fx.Int32(32) for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
                npk = [
                    [
                        (ng[gu][ni] // fx.Int32(16)) % fx.Int32(2)
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
            # per column tile one vector address (lane*16 B + block base + the K-wave's
            # K start); the K position is (kt//4)*4096 in an SGPR + (kt%4)*1024 imm
            wvo = [
                [
                    lane * fx.Int32(4)
                    + nblk[gu][ni] * fx.Int32(KT * 256)
                    + wave_k * fx.Int32(KTW * 256)
                    for ni in range_constexpr(NI)
                ]
                for gu in range_constexpr(2)
            ]
            svo = [
                [
                    lane
                    + mni[gu][ni] * fx.Int32(SC_STRIDE_N0)
                    + wave_k * fx.Int32(KTW // 2 * 64)
                    for ni in range_constexpr(NI)
                ]
                for gu in range_constexpr(2)
            ]

            def load_b_tile(kt, prev):
                so = _sconst((kt // 4) * 4096)
                bb = [
                    [
                        bop.buffer_load(
                            wr,
                            wvo[gu][ni] + fx.Int32((kt % 4) * 256),
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=b_cache_mod,
                            soffset_bytes=so,
                        )
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                if const_expr(kt % 2 == 0):
                    sso = _sconst((kt // 8) * 1024)  # one scale dword per 2 K-tiles
                    sc = [
                        [
                            bop.buffer_load(
                                sr,
                                svo[gu][ni] + fx.Int32(((kt // 2) % 4) * 64),
                                vec_width=1,
                                dtype=fx.Int32,
                                soffset_bytes=sso,
                            )
                            for ni in range_constexpr(NI)
                        ]
                        for gu in range_constexpr(2)
                    ]
                else:
                    sc = prev[1]
                return bb, sc

            vec2_bf16 = ir.Type.parse("vector<2xbf16>")

            def upconvert(raw4, ku, scale_f32):
                # raw4[ku]: 8 fp4 of K-step ku -> 4 x cvt_scalef32_pk_bf16_fp4 -> v8bf16
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
            acc = [
                [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(NI)]
                for _ in range(2)
            ]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for gu in range_constexpr(2):
                for ni in range_constexpr(NI):
                    acc[gu][ni].store(zero4)

            def _frag(v8):
                t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                t.store(v8)
                return t

            def compute_tile(bb, sc, aa, kt):
                a_t = [_frag(aa[ku]) for ku in range_constexpr(4)]
                for gu in range_constexpr(2):
                    for ni in range_constexpr(NI):
                        s = _e8m0_byte_to_f32(
                            fx.Int32(sc[gu][ni]), fx.Int32((kt % 2) * 2) + npk[gu][ni]
                        )
                        raw4 = fx.Vector(bb[gu][ni])
                        for ku in range_constexpr(4):
                            fx.gemm(
                                mma_atom,
                                acc[gu][ni],
                                a_t[ku],
                                _frag(upconvert(raw4, ku, s)),
                                acc[gu][ni],
                            )

            # pipeline: batch 0 -> LDS, W ring; per tile: (batch loads) W load, LDS
            # read, MFMAs, scheduling barrier
            abuf = load_a_batch(0)
            ring = []
            for t in range_constexpr(prefetch):
                ring.append(load_b_tile(t, ring[-1] if ring else None))
            stage_a_batch(abuf, 0)
            abuf = None
            for kt in range_constexpr(KTW):
                if const_expr(kt % KB == 0 and kt + KB < KTW):
                    abuf = load_a_batch(kt // KB + 1)  # before this iteration's W loads
                if const_expr(kt + prefetch < KTW):
                    ring.append(load_b_tile(kt + prefetch, ring[-1]))
                bb, sc = ring.pop(0)
                compute_tile(bb, sc, read_a_tile(kt), kt)
                rocdl.sched_barrier(0)
                if const_expr(kt % KB == KB - 1 and kt + 1 < KTW):
                    stage_a_batch(
                        abuf, (kt // KB + 1) % 2
                    )  # this batch's reads are done
                    abuf = None

            if const_expr(KW > 1):
                # K-reduce: K-wave 1 parks its partial sums in the (now free) A slots,
                # K-wave 0 adds them and runs the epilogue alone
                s_waitcnt_lgkm0()
                gpu.barrier()
                red = [
                    [
                        (
                            (wave_n * fx.Int32(2) + fx.Int32(gu)) * fx.Int32(NI)
                            + fx.Int32(ni)
                        )
                        * fx.Int32(1024)
                        + lane * fx.Int32(16)
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                if wave_k > fx.Int32(0):
                    for gu in range_constexpr(2):
                        for ni in range_constexpr(NI):
                            ptr = bop.get_element_ptr(
                                lds,
                                byte_offset=fx.as_ir_value(red[gu][ni]),
                                elem_type=T.i8,
                            )
                            llvm.StoreOp(
                                fx.as_ir_value(
                                    fx.Vector(fx.memref_load_vec(acc[gu][ni]))
                                ),
                                ptr,
                                alignment=16,
                            )
                s_waitcnt_lgkm0()
                gpu.barrier()
                if wave_k == fx.Int32(0):
                    for gu in range_constexpr(2):
                        for ni in range_constexpr(NI):
                            v = fx.Vector(fx.memref_load_vec(acc[gu][ni]))
                            ptr = bop.get_element_ptr(
                                lds,
                                byte_offset=fx.as_ir_value(red[gu][ni]),
                                elem_type=T.i8,
                            )
                            pv = fx.Vector(
                                llvm.load(T.vec(4, T.f32), ptr, alignment=16)
                            )
                            acc[gu][ni].store(
                                fx.Vector.from_elements(
                                    [v[i] + pv[i] for i in range_constexpr(4)],
                                    fx.Float32,
                                )
                            )

            # epilogue: swigluoai(gate, up) -> bf16 [sorted_row, inter], padding masked
            neg_limit = -fx.Float32(f32_limit)
            alpha = fx.Float32(f32_alpha)

            def epilogue():
                for ii in range_constexpr(4):
                    sorted_row = mbase + q16 * fx.Int32(4) + fx.Int32(ii)
                    valid = ep_tok[ii] < i32_ntok
                    for ni in range_constexpr(NI):
                        g = fx.Float32(fx.Vector(fx.memref_load_vec(acc[0][ni]))[ii])
                        u = fx.Float32(fx.Vector(fx.memref_load_vec(acc[1][ni]))[ii])
                        yb = _swigluoai_f32(g, u, alpha, neg_limit).to(fx.BFloat16)
                        out_idx = (
                            sorted_row * fx.Int32(INTER)
                            + nbase
                            + fx.Int32(ni * 16)
                            + l16
                        )
                        bop.buffer_store(yb, outr, _raw(out_idx), mask=valid)

            if const_expr(KW > 1):
                if wave_k == fx.Int32(0):
                    epilogue()
            else:
                epilogue()

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        f32_alpha: fx.Float32,
        f32_limit: fx.Float32,
        arg_out: fx.Int64,
        arg_zero: fx.Int64,
        i32_zero_dw: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = fx.Int64(i32_grid)
        kernel(
            arg_x,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            f32_alpha,
            f32_limit,
            arg_out,
            arg_zero,
            i32_zero_dw,
            **(
                {"value_attrs": {"rocdl.waves_per_eu": waves_per_eu}}
                if waves_per_eu
                else {}
            ),
        ).launch(grid=(grid_x, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    return launch
