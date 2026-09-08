# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode gate/up GEMM (bf16 x MXFP8 W13, MFMA 16x16x32) fused with swiglu-OAI.

The a16w4 decode gemm1 with fp8 weights (``moe_a16w4_decode/gemm1.py`` explains
the workgroup / wave split, the A staging through LDS and the W ring; only what
the 1-byte elements change is described here):

* W tile = one 1 KB preshuffled block = 16 columns x 64 K (fp4: 128 K). Lane
  (klane, n) holds K ``klane*16 .. +16`` of column n: two MFMA K-steps of 8 per
  tile, unpacked right before the MFMA with ``v_cvt_scalef32_pk_bf16_fp8``.
* A is still staged in 256 B chunks (128 K of a row = two W tiles); the MFMA A
  fragment for tile ``kt``, K-step ``ku`` is row ``l16``, K ``(kt%2)*64 + q16*16 +
  ku*8`` of that chunk.
* the per-32 e8m0 scales: a lane's 16 K lie in 32-K group ``(kt%2)*2 + klane//2``
  of the 128-K tile, so per 256 K it loads two scale dwords (groups
  ``klane//2`` and ``2 + klane//2``); each dword packs gate/up (bytes 0/1) for
  the two 128-K tiles of the 256 K (bytes +0 / +2).

Layouts (aiter ``shuffle_weight(is_guinterleave=True, gate_up=True)`` +
``shuffle_scale(..., True, True)``, what vLLM's ``shuffle_mxfp8_moe_weights`` stores):
  W13     [E, I/16, 2 (gate, up), K/64, klane 4, nlane 16, 16 B]  fp8 e4m3
  W13_sc  [E, I/16, K/256, klane 4, nlane 16] dwords  e8m0 (bytes: kt, gu)
Routing (``inline_sort`` / ``sort_decode``) and the epilogue are the a16w4 ones.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from vllm.models.minimax_m3.amd.ops.moe_a16w4_decode.utils import (
    _e8m0_byte_to_f32,
    _global_i32_ptr,
    _swigluoai_f32,
    inline_sort_max_pairs,
    inline_sort_table,
)

from .utils import _fp8x8_to_bf16

BM = 16  # rows per m-block (one MFMA M tile)
NW = 4  # waves per workgroup
LDS_PAD = 16  # bytes of padding per LDS row: conflict-free 16 B reads
TILE_K = 64  # K per W tile (one 1 KB block per 16 columns)
CHUNK_K = 128  # K per A load chunk (256 B of bf16 per row, one dwordx4 per lane)
# Tiles from the a16w4 MI355X sweeps (same A/LDS traffic, W tiles half the K): up to
# LARGE_M_TOKENS two N-waves x two K-waves of 16 columns (TILE_N 32); above, four
# N-waves of 16 columns (TILE_N 64) with 3 waves per EU. A batches of 2 chunks,
# W ring 3 tiles deep, non-temporal W.
LARGE_M_TOKENS = 128


def compile_gemm1(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    n_tokens,
    inline_sort=False,
    large_m=None,
    prefetch=None,
    waves_per_eu=None,
):
    """Kernel for batches of up to ``n_tokens`` tokens: only the tile choice and the
    inline-sort scan length depend on it, so different ``n_tokens`` often give the
    same kernel (``launch.kernel_name``). ``launch.tile_n`` is the N tile for the
    grid. ``large_m`` / ``prefetch`` / ``waves_per_eu`` override the defaults for
    lab sweeps only (the kernel name carries them)."""
    if large_m is None:
        large_m = n_tokens > LARGE_M_TOKENS
    TILE_N, KW, wpe = (64, 1, 3) if large_m else (32, 2, None)
    if waves_per_eu is None:
        waves_per_eu = wpe
    KB = 2  # A chunks per batch through LDS
    if prefetch is None:
        prefetch = 3  # W tiles in flight
    b_cache_mod = 2  # non-temporal W loads
    K, INTER = D_HIDDEN, D_INTER
    N_OUT = 2 * INTER
    NWN = NW // KW  # N-waves
    NPW = TILE_N // NWN  # columns per wave (gate and up each)
    NI = NPW // 16
    TPC = CHUNK_K // TILE_K  # W tiles per A chunk
    BT = KB * TPC  # W tiles per A batch
    assert NPW % 16 == 0 and INTER % TILE_N == 0 and K % (KW * KB * CHUNK_K) == 0
    KT = K // TILE_K  # W tiles
    NNB = INTER // TILE_N  # N blocks per m-block
    KTW = KT // KW  # W tiles per K-wave
    KCW = (K // CHUNK_K) // KW  # A chunks per K-wave
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
    W_BYTES = NE * N_OUT * K
    SC_K1 = K // 256  # scale dwords per lane per column group
    SC_STRIDE_N0 = SC_K1 * 64
    SW_BYTES = NE * N_OUT * (SC_K1 * 8)
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # A slots / K-reduce scratch
        tab: fx.Array[fx.Int32, 32]  # routing table (inline sort only; 128 B)

    name = (
        f"m3_gemm1_a16w8_h{K}_i{INTER}_ne{NE}_tn{TILE_N}_kw{KW}_kb{KB}_pf{prefetch}"
        f"_bcm{b_cache_mod}"
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
                # batch b of every K-wave: chunks kw*KCW + b*KB + j; base in an SGPR,
                # j*256 in the immediate offset field
                out = []
                for kw in range_constexpr(KW):
                    so = (kw * KCW + b * KB) * 256
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

            # MFMA A fragment of W tile kt, K-step ku: row l16, chunk kt//2, K
            # (kt%2)*64 + q16*16 + ku*8 -> 16 B tile (kt%2)*8 + q16*2 + ku of the chunk
            rd_tile = wave_k * KSLOT_T + l16 * RS_T + q16 * 2

            def read_a(kt, ku):
                c = kt // TPC
                slot, j, b = (c // KB) % 2, c % KB, kt % TPC
                return lds_load16(
                    rd_tile + (slot * SLOT_T + j * 16 + b * 8 + ku)
                ).bitcast(fx.BFloat16)

            # W addressing (gu 0 = gate, 1 = up; gate/up 16-row blocks interleaved), NI
            # 16-column tiles per wave; scale dword packs gate (byte 0/2) and up (1/3)
            nbase = nb * TILE_N + wave_n * NPW
            n0 = [(nbase + ni * 16) // 16 for ni in range_constexpr(NI)]
            nblk = [
                [e * (N_OUT // 16) + n0[ni] * 2 + gu for ni in range_constexpr(NI)]
                for gu in range_constexpr(2)
            ]
            mni = [e * (N_OUT // 32) + n0[ni] for ni in range_constexpr(NI)]
            # per column tile one vector address (lane*16 B + block base + the K-wave's
            # K start); the K position is (kt//4)*4096 in an SGPR + (kt%4)*1024 imm
            wvo = [
                [
                    lane * 4 + nblk[gu][ni] * (KT * 256) + wave_k * (KTW * 256)
                    for ni in range_constexpr(NI)
                ]
                for gu in range_constexpr(2)
            ]
            # scale dwords of 32-K groups klane//2 (b = 0) and 2 + klane//2 (b = 1)
            lane_sc = [(2 * b + q16 // 2) * 16 + l16 for b in range_constexpr(2)]
            svo = [
                [
                    lane_sc[b] + mni[ni] * SC_STRIDE_N0 + wave_k * (KTW // 4 * 64)
                    for ni in range_constexpr(NI)
                ]
                for b in range_constexpr(2)
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
                if const_expr(kt % 4 == 0):
                    g1 = kt // 4  # 256-K group: two scale dwords per lane
                    sc = [
                        [
                            fx.Int32(
                                buffer_ops.buffer_load(
                                    sr,
                                    svo[b][ni] + (g1 % 4) * 64,
                                    vec_width=1,
                                    dtype=fx.Int32,
                                    soffset_bytes=(g1 // 4) * 1024,
                                )
                            )
                            for ni in range_constexpr(NI)
                        ]
                        for b in range_constexpr(2)
                    ]
                else:
                    sc = prev[1]
                return bb, sc

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

            def compute_tile(bb, sc, kt):
                a_t = [_frag(read_a(kt, ku)) for ku in range_constexpr(TPC)]
                # scale byte: 128-K tile (kt//2)%2 of the 256-K group -> +2, gate/up -> +gu
                for gu in range_constexpr(2):
                    for ni in range_constexpr(NI):
                        s = _e8m0_byte_to_f32(sc[kt % 2][ni], gu + ((kt // 2) % 2) * 2)
                        for ku in range_constexpr(TPC):
                            b8 = _fp8x8_to_bf16(
                                bb[gu][ni][2 * ku], bb[gu][ni][2 * ku + 1], s
                            )
                            fx.gemm(mma_atom, acc[gu][ni], a_t[ku], _frag(b8), acc[gu][ni])

            # pipeline: batch 0 -> LDS, W ring; per tile: (batch loads) W load, LDS
            # read, MFMAs
            abuf = load_a_batch(0)
            ring = []
            for t in range_constexpr(prefetch):
                ring.append(load_b_tile(t, ring[-1] if ring else None))
            stage_a_batch(abuf, 0)
            abuf = None
            for kt in range_constexpr(KTW):
                if const_expr(kt % BT == 0 and kt + BT < KTW):
                    abuf = load_a_batch(kt // BT + 1)  # before this iteration's W loads
                if const_expr(kt + prefetch < KTW):
                    ring.append(load_b_tile(kt + prefetch, ring[-1]))
                bb, sc = ring.pop(0)
                compute_tile(bb, sc, kt)
                if const_expr(kt % BT == BT - 1 and kt + 1 < KTW):
                    stage_a_batch(abuf, (kt // BT + 1) % 2)  # this batch's reads are done
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
