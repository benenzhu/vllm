# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode gate/up GEMM (bf16 x MXFP4 W13, MFMA 16x16x32) fused with swiglu-OAI.

One workgroup per (16-row m-block, TILE_N-column n-block). Its 4 waves split N
(``large_m``: four N-waves of 16 columns; otherwise two N-waves x two K-waves
with an LDS reduce at the end) and share the A rows: the workgroup loads one
batch of two 128-K tiles of the 16 rows (one dwordx4 per lane per K-tile, each
instruction 4 rows x 256 B = 8 full cache lines) into LDS and every wave reads
its MFMA A fragments from there. Two LDS slots, one barrier per batch; the batch
loads go out before the same iteration's W loads so the in-order vmcnt wait for
that W tile also covers them. W streams through a VGPR ring three tiles deep
with non-temporal loads (1 KB contiguous per wave instruction in aiter's
preshuffled layout), the per-32 e8m0 scales come one packed dword per 256 K, and
the fp4 -> bf16 conversion runs right before each MFMA.

Routing: ``inline_sort`` (n_tokens <= 16) has no sort kernel: each block derives
its expert and rows from ``topk_ids`` with a wave ballot
(``utils.inline_sort_table``) and the blocks of routing pair 0 zero the stage-2
output; otherwise the rows come expert-sorted from ``sort_decode``. Padding rows
carry a token id >= n_tokens: their A loads read 0 through the OOB-clamped buffer
resource and their outputs are masked.

The global loads and the masked epilogue store go through ``buffer_ops`` (raw
buffer instructions) because the layout copy API has no way to put the K
position into the soffset SGPR or to mask a scalar store; LDS is a 16 B tile
view of the shared struct and uses ``fx.copy``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .utils import (
    _e8m0_byte_to_f32,
    _global_i32_ptr,
    _swigluoai_f32,
    inline_sort_max_pairs,
    inline_sort_table,
)

BM = 16  # rows per m-block (one MFMA M tile)
NW = 4  # waves per workgroup
LDS_PAD = 16  # bytes of padding per LDS row: conflict-free 16 B reads
# Tiles from the MI355X sweeps (module docstring): up to LARGE_M_TOKENS two N-waves x
# two K-waves of 16 columns (TILE_N 32); above, four N-waves of 16 columns (TILE_N 64)
# with 3 waves per EU. Both: A batches of 2 K-tiles, W ring 3 deep, non-temporal W.
LARGE_M_TOKENS = 128


def compile_gemm1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    n_tokens,
    w_layout="standard",
    inline_sort=False,
):
    """Kernel for batches of up to ``n_tokens`` tokens: only the tile choice and the
    inline-sort scan length depend on it, so different ``n_tokens`` often give the
    same kernel (``launch.kernel_name``). ``launch.tile_n`` is the N tile for the
    grid."""
    large_m = n_tokens > LARGE_M_TOKENS
    TILE_N, KW, waves_per_eu = (64, 1, 3) if large_m else (32, 2, None)
    KB = 2  # 128-K tiles per A batch through LDS
    prefetch = 3  # W tiles in flight
    b_cache_mod = 2  # non-temporal W loads
    K, INTER = D_HIDDEN, D_INTER
    N_OUT = 2 * INTER
    assert w_layout in ("standard", "guinterleave")
    NWN = NW // KW  # N-waves
    NPW = TILE_N // NWN  # columns per wave (gate and up each)
    NI = NPW // 16
    assert NPW % 16 == 0 and INTER % TILE_N == 0 and K % 256 == 0
    KT = K // 128  # 128-K tiles: 1 KB of W per 16 columns, 256 B of A per row
    KTW = KT // KW  # K-tiles per K-wave
    NNB = INTER // TILE_N
    ROWB = KB * 256  # A bytes per row per batch (per K-wave)
    RS = ROWB + LDS_PAD  # LDS row stride
    KSLOT = BM * RS  # one K-wave's batch
    SLOT = KW * KSLOT
    RED_BYTES = (KW - 1) * NWN * 2 * NI * 1024  # K-reduce scratch (reuses the A slots)
    LDS_BYTES = max(2 * SLOT, RED_BYTES)
    # LDS is addressed in 16 B tiles below
    RS_T, KSLOT_T, SLOT_T = RS // 16, KSLOT // 16, SLOT // 16
    if inline_sort:
        assert n_tokens <= BM, "inline sort: every expert's rows fit one m-block"
        max_pairs = inline_sort_max_pairs(n_tokens, TOPK, BM)
    # W (mxfp4) preshuffle layout (aiter make_preshuffle_b_layout, N-major, fp4 bytes):
    # (N_OUT/16, K/128, klane 4, nlane 16, kpack 16 B): one 16-col x 128-K block is 1 KB
    # contiguous; lane (klane, n) holds the 32 K of column n at klane -> lane*16 B.
    W_BYTES = NE * N_OUT * (K // 2)
    # scale preshuffle (make_preshuffle_scale_layout, e8m0 u8): (N_OUT/32, K/256,
    # klane 4, nlane 16) dwords; one dword = 2 K-tiles x 2 N-halves (16 cols each).
    SC_K1 = K // 256
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # A slots / K-reduce scratch
        tab: fx.Array[fx.Int32, 32]  # routing table (inline sort only; 128 B)

    name = (
        f"m3_gemm1_a16w4_h{K}_i{INTER}_ne{NE}_tn{TILE_N}_kw{KW}_kb{KB}_pf{prefetch}"
        f"_bcm{b_cache_mod}"
        + ("" if w_layout == "standard" else "_gu")
        + (f"_w{waves_per_eu}" if waves_per_eu else "")
        + (f"_isort{max_pairs}" if inline_sort else "")
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
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        wave_n, wave_k = wave % NWN, wave // NWN
        mb, nb = pid // NNB, pid % NNB
        mbase = mb * BM

        # LDS as 16 B tiles; one dwordx4 per lane per copy
        lds16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.a.ptr), fx.make_layout(LDS_BYTES // 4, 1)
            ),
            fx.make_layout(4, 1),
        )
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def lds_store16(tile, vec4):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            r.store(vec4)
            fx.copy(lds_atom, r, fx.slice(lds16, (None, tile)))

        def lds_load16(tile):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
            return r.load()

        if const_expr(inline_sort):
            # block = routing pair mb: expert + rows from a ballot over the pairs
            tab = smem.tab.ptr
            e_pair, owner, _nrows, build_tab = inline_sort_table(
                arg_mind, i32_ntok, TOPK, mb, lane, tab, max_pairs=max_pairs
            )
            if owner:
                build_tab()
            cumsum0 = i32_ntok * (TOPK * BM)
            go = owner
            # zero the stage-2 output (gemm2 accumulates with atomics): the NNB blocks
            # of pair 0 stride over it, one dword per thread
            if mb == 0:
                zb = _global_i32_ptr(arg_zero)
                for iv in range(pid * (64 * NW) + tx, i32_zero_dw, NNB * 64 * NW):
                    zb[fx.Int32(iv)] = fx.Int32(0)

            def mind_at(row):
                return fx.Int32(tab[row])

        else:
            mind = _global_i32_ptr(arg_mind)
            cumsum0 = fx.Int32(_global_i32_ptr(arg_cumsum)[0])
            go = mbase < cumsum0

            def mind_at(row):
                return fx.Int32(mind[mbase + row])

        if go:
            if const_expr(inline_sort):
                e = e_pair
            else:
                e = fx.Int32(
                    fx.rocdl.readfirstlane(
                        T.i32, fx.Int32(_global_i32_ptr(arg_eids)[mb])
                    )
                )
            # A staging: wave w loads row w*4 + lane//16, 16 B chunk j*16 + lane%16
            ld_row = wave * 4 + q16
            ld_tok = mind_at(ld_row) & 0xFFFFFF
            # epilogue rows: lane (q16, l16) holds rows q16*4 + ii of column l16
            ep_tok = [mind_at(q16 * 4 + ii) & 0xFFFFFF for ii in range_constexpr(4)]
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x, num_records_bytes=fx.Int64(i32_ntok) * (K * 2)
            )
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_bq, num_records_bytes=W_BYTES
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_bscale, num_records_bytes=SW_BYTES
            )
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=fx.Int64(cumsum0) * (INTER * 2)
            )
            ld_gdw = (ld_tok * (K * 2) + l16 * 16) // 4
            ld_tile = ld_row * RS_T + l16

            def load_a_batch(b):
                # batch b of every K-wave: tiles kw*KTW + b*KB + j; base in an SGPR,
                # j*256 in the immediate offset field
                out = []
                for kw in range_constexpr(KW):
                    so = (kw * KTW + b * KB) * 256
                    out += [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                xr,
                                ld_gdw + j * 64,
                                vec_width=4,
                                dtype=fx.Int32,
                                soffset_bytes=so,
                            )
                        )
                        for j in range_constexpr(KB)
                    ]
                return out

            def stage_a_batch(regs, slot):
                for kw in range_constexpr(KW):
                    for j in range_constexpr(KB):
                        lds_store16(
                            ld_tile + (slot * SLOT_T + kw * KSLOT_T + j * 16),
                            regs[kw * KB + j],
                        )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

            # MFMA A fragment (K-step ku of tile kt): row l16, K = klane*32 + ku*8
            rd_tile = wave_k * KSLOT_T + l16 * RS_T + q16 * 4

            def read_a_tile(kt):
                slot, j = (kt // KB) % 2, kt % KB
                return [
                    lds_load16(rd_tile + (slot * SLOT_T + j * 16 + ku)).bitcast(
                        fx.BFloat16
                    )
                    for ku in range_constexpr(4)
                ]

            # W addressing (gu 0 = gate, 1 = up), NI 16-column tiles per wave
            nbase = nb * TILE_N + wave_n * NPW
            if const_expr(w_layout == "guinterleave"):
                # gate/up 16-row blocks interleaved; the scale dword packs gate
                # (bytes 0/2) and up (1/3)
                n0 = [(nbase + ni * 16) // 16 for ni in range_constexpr(NI)]
                nblk = [
                    [e * (N_OUT // 16) + n0[ni] * 2 + gu for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
                mni = [
                    [e * (N_OUT // 32) + n0[ni] for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
                npk = [
                    [fx.Int32(gu) for _ in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
            else:
                ng = [
                    [
                        e * N_OUT + nbase + (ni * 16 + gu * INTER)
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                nblk = [
                    [ng[gu][ni] // 16 for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
                mni = [
                    [ng[gu][ni] // 32 for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
                npk = [
                    [(ng[gu][ni] // 16) % 2 for ni in range_constexpr(NI)]
                    for gu in range_constexpr(2)
                ]
            # per column tile one vector address (lane*16 B + block base + the K-wave's
            # K start); the K position is (kt//4)*4096 in an SGPR + (kt%4)*1024 imm
            wvo = [
                [
                    lane * 4 + nblk[gu][ni] * (KT * 256) + wave_k * (KTW * 256)
                    for ni in range_constexpr(NI)
                ]
                for gu in range_constexpr(2)
            ]
            svo = [
                [
                    lane + mni[gu][ni] * SC_STRIDE_N0 + wave_k * (KTW // 2 * 64)
                    for ni in range_constexpr(NI)
                ]
                for gu in range_constexpr(2)
            ]

            def load_b_tile(kt, prev):
                so = (kt // 4) * 4096
                bb = [
                    [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                wr,
                                wvo[gu][ni] + (kt % 4) * 256,
                                vec_width=4,
                                dtype=fx.Int32,
                                cache_modifier=b_cache_mod,
                                soffset_bytes=so,
                            )
                        )
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                if const_expr(kt % 2 == 0):
                    sso = (kt // 8) * 1024  # one scale dword per 2 K-tiles
                    sc = [
                        [
                            fx.Int32(
                                buffer_ops.buffer_load(
                                    sr,
                                    svo[gu][ni] + ((kt // 2) % 4) * 64,
                                    vec_width=1,
                                    dtype=fx.Int32,
                                    soffset_bytes=sso,
                                )
                            )
                            for ni in range_constexpr(NI)
                        ]
                        for gu in range_constexpr(2)
                    ]
                else:
                    sc = prev[1]
                return bb, sc

            v2bf16 = T.vec(2, T.bf16)

            def upconvert(raw4, ku, scale):
                # raw4[ku]: 8 fp4 of K-step ku -> 4 x cvt_scalef32_pk_bf16_fp4 -> v8bf16
                halves = [
                    fx.rocdl.cvt_scalef32_pk_bf16_fp4(v2bf16, raw4[ku], scale, sel)
                    for sel in range_constexpr(4)
                ]
                return fx.Vector.from_elements(
                    [fx.Vector(h).bitcast(fx.Int32)[0] for h in halves], fx.Int32
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
                        s = _e8m0_byte_to_f32(sc[gu][ni], npk[gu][ni] + (kt % 2) * 2)
                        for ku in range_constexpr(4):
                            fx.gemm(
                                mma_atom,
                                acc[gu][ni],
                                a_t[ku],
                                _frag(upconvert(bb[gu][ni], ku, s)),
                                acc[gu][ni],
                            )

            # pipeline: batch 0 -> LDS, W ring; per tile: (batch loads) W load, LDS
            # read, MFMAs
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
                if const_expr(kt % KB == KB - 1 and kt + 1 < KTW):
                    stage_a_batch(
                        abuf, (kt // KB + 1) % 2
                    )  # this batch's reads are done
                    abuf = None

            if const_expr(KW > 1):
                # K-reduce: K-wave 1 parks its partial sums in the (now free) A slots,
                # K-wave 0 adds them and runs the epilogue alone
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                red = [
                    [
                        ((wave_n * 2 + gu) * NI + ni) * 64 + lane
                        for ni in range_constexpr(NI)
                    ]
                    for gu in range_constexpr(2)
                ]
                if wave_k > 0:
                    for gu in range_constexpr(2):
                        for ni in range_constexpr(NI):
                            lds_store16(
                                red[gu][ni], acc[gu][ni].load().bitcast(fx.Int32)
                            )
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                if wave_k == 0:
                    for gu in range_constexpr(2):
                        for ni in range_constexpr(NI):
                            v = acc[gu][ni].load()
                            pv = lds_load16(red[gu][ni]).bitcast(fx.Float32)
                            acc[gu][ni].store(
                                fx.Vector.from_elements(
                                    [v[i] + pv[i] for i in range_constexpr(4)],
                                    fx.Float32,
                                )
                            )

            # epilogue: swigluoai(gate, up) -> bf16 [sorted_row, inter], padding masked
            neg_limit = -f32_limit

            def epilogue():
                for ii in range_constexpr(4):
                    sorted_row = mbase + q16 * 4 + ii
                    valid = ep_tok[ii] < i32_ntok
                    for ni in range_constexpr(NI):
                        g = acc[0][ni].load()[ii]
                        u = acc[1][ni].load()[ii]
                        yb = _swigluoai_f32(g, u, f32_alpha, neg_limit).to(fx.BFloat16)
                        out_idx = sorted_row * INTER + nbase + ni * 16 + l16
                        buffer_ops.buffer_store(yb, outr, out_idx, mask=valid)

            if const_expr(KW > 1):
                if wave_k == 0:
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

    launch.kernel_name = name
    launch.tile_n = TILE_N
    return launch
