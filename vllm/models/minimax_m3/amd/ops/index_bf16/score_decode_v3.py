# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode index scorer, v3: row-contiguous K loads through a per-wave LDS transpose.

Contract as ``score_decode`` (the score phase of ``minimax_m3_index_decode``). Each
wave owns a run of one request's blocks (wave-parallel work split: one lane per
request, wave prefix sums) and streams them in 32-token pieces (8 KB): eight
1 KB-contiguous loads per piece (a wave instruction covers four 256 B token rows,
so the request count is a quarter of the direct A-fragment layout), written to
the wave's own padded LDS slot (272 B row stride) and read back as MFMA A
fragments; no workgroup barrier, the LDS region is private to the wave. Two
pieces stay in flight across the loop (loop-carried registers), the next two
blocks' pages are loaded one iteration ahead. The <= 16 query rows of the
request are the MFMA B operand in 16 VGPRs.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from .utils import _global_i32_ptr, _maxf_nn, _wave_prefix_sum_i32, _xlane_max4_pair

NW = 4
WAVES = int(os.environ.get("M3_IDX_DWAVES", "1024"))  # graph-constant wave budget (1024 measured best: 4-5 blocks per wave at 800K conc 1)
BLK = 128
D = 128
BLOCK_BYTES = BLK * D * 2
KU = D // 32
PIECE_TOK = 32  # tokens per piece (2 M-tiles)
PIECES = BLK // PIECE_TOK
PIECE_LOADS = PIECE_TOK * D * 2 // 1024  # 1 KB per wave instruction
RING = int(os.environ.get("M3_IDX_DRING", "2"))  # pieces in flight (lab knob)
LDS_PAD = 16
RS = D * 2 + LDS_PAD
RS_T = RS // 16
SLOT_T = PIECE_TOK * RS_T  # 16 B tiles per piece slot
WAVE_T = RING * SLOT_T
LDS_BYTES = NW * WAVE_T * 16
MAX_REQS = 64
_KNOBS = set(filter(None, os.environ.get("M3_IDX_KNOBS", "").split(",")))
K_CACHE_MOD = 0 if "nont" in _KNOBS else 2


def compile_score_decode_v3():
    name = f"m3_index_score_decode_bf16_v3_w{WAVES}_ring{RING}" + "".join(
        "_" + k for k in sorted(_KNOBS)
    )

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nreq: fx.Int32,
        i32_qlen: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_total_q: fx.Int32,
        i64_kv_bytes: fx.Int64,
    ):
        k_cache_mod = K_CACHE_MOD  # closure scalar of load_piece: part of the cache key
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        wid = pid * NW + wave
        seq = _global_i32_ptr(arg_seq)
        bt = _global_i32_ptr(arg_bt)

        # work split, one lane per request
        has = lane < i32_nreq
        L_l = fx.Int32(seq[has.select(lane, fx.Int32(0))])
        L_l = (has & (L_l > 0)).select(L_l, fx.Int32(0))
        nb_l = (L_l + (BLK - 1)) // BLK
        incl_nb = _wave_prefix_sum_i32(nb_l, lane)
        total_blocks = fx.Int32(fx.rocdl.readlane(T.i32, incl_nb, 63))
        bpw = (total_blocks + (WAVES - 1)) // WAVES
        bpw = (bpw > 0).select(bpw, fx.Int32(1))
        nw_l = (nb_l + bpw - 1) // bpw
        incl_nw = _wave_prefix_sum_i32(nw_l, lane)
        off_l = incl_nw - nw_l
        owns = has & (wid >= off_l) & (wid < off_l + nw_l)
        own_mask = fx.Int64(fx.rocdl.ballot(T.i64, owns))
        r = fx.Int64(fx.math.cttz(own_mask)).to(fx.Int32)
        r = (own_mask != 0).select(r, fx.Int32(-1))
        r_ok = (own_mask != 0).select(r, fx.Int32(0))
        my_w = wid - fx.Int32(fx.rocdl.readlane(T.i32, off_l, r_ok))
        nb = fx.Int32(fx.rocdl.readlane(T.i32, nb_l, r_ok))
        L = fx.Int32(fx.rocdl.readlane(T.i32, L_l, r_ok))
        if r >= 0:
            blk0 = my_w * bpw
            blk1e = blk0 + bpw
            blk1 = (blk1e < nb).select(blk1e, nb)
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=fx.Int64(i32_total_q) * (D * 2)
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_score, num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4
            )
            # B fragments (query rows l16): lane (l16, q16) at K-step ku holds dims
            # q16*32 + ku*8 .. +8, the same dim permutation as the LDS A fragments
            row = r * i32_qlen + l16
            qf = [
                fx.Vector(
                    buffer_ops.buffer_load(
                        qr,
                        (row * (D * 2) + q16 * 64 + ku * 16) // 4,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                ).bitcast(fx.BFloat16)
                for ku in range_constexpr(KU)
            ]
            qpos = L - i32_qlen + l16
            kv_len = qpos + 1
            kv_len = (kv_len > 0).select(kv_len, fx.Int32(0))
            nb_row = (kv_len + (BLK - 1)) // BLK
            loc0 = nb_row - i32_local
            loc0 = (loc0 > 0).select(loc0, fx.Int32(0))
            bt_row = r * i32_bt_stride
            last_blk = nb - 1

            def page_of(blk):
                bi = (blk < nb).select(blk, last_blk)
                return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(bt[bt_row + bi])))

            # LDS: this wave's RING piece slots, 16 B tiles
            lds16 = fx.logical_divide(
                fx.make_view(
                    fx.recast_iter(fx.Int32, smem.a.ptr), fx.make_layout(LDS_BYTES // 4, 1)
                ),
                fx.make_layout(4, 1),
            )
            lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            wave_t = wave * WAVE_T

            def lds_store16(tile, vec4):
                t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                t.store(vec4)
                fx.copy(lds_atom, t, fx.slice(lds16, (None, tile)))

            def lds_load16(tile):
                t = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(lds_atom, fx.slice(lds16, (None, tile)), t)
                return t.load()

            # row-contiguous loads: instruction i of a piece covers rows 4i .. 4i+3,
            # lane (row 4i + lane//16, chunk lane%16); page and piece in soffset
            row_off = ((lane // 16) * (D * 2) + (lane % 16) * 16) // 4
            st_tiles = [
                (4 * i + lane // 16) * RS_T + lane % 16 for i in range_constexpr(PIECE_LOADS)
            ]

            def load_piece(page, p):
                # one 32 KB buffer resource per block (64-bit base: the index cache may
                # exceed 4 GB); piece and instruction offsets are constants in soffset
                kr = buffer_ops.create_buffer_resource_from_addr(
                    arg_kv + fx.Int64(page) * BLOCK_BYTES, num_records_bytes=BLOCK_BYTES
                )
                return [
                    fx.Vector(
                        buffer_ops.buffer_load(
                            kr,
                            row_off,
                            vec_width=4,
                            dtype=fx.Int32,
                            cache_modifier=k_cache_mod,
                            soffset_bytes=p * (PIECE_TOK * D * 2) + i * 1024,
                        )
                    )
                    for i in range_constexpr(PIECE_LOADS)
                ]

            def stage_piece(regs, slot):
                for i in range_constexpr(PIECE_LOADS):
                    lds_store16(wave_t + slot * SLOT_T + st_tiles[i], regs[i])

            mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            neg_inf = fx.Float32(float("-inf"))
            f_local = fx.Float32(1e29)
            f_init = fx.Float32(1e30)
            rd_base = l16 * RS_T + q16 * 4

            def frag8(v8):
                t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                t.store(v8)
                return t

            def mtile(slot, mt_in_piece, blk, mt, run):
                a = [
                    lds_load16(
                        wave_t + slot * SLOT_T + mt_in_piece * 16 * RS_T + rd_base + ku
                    ).bitcast(fx.BFloat16)
                    for ku in range_constexpr(KU)
                ]
                acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                acc.store(zero4)
                for ku in range_constexpr(KU):
                    fx.gemm(mma, acc, frag8(a[ku]), frag8(qf[ku]), acc)
                v = acc.load()
                vis = qpos - (blk * BLK + mt * 16 + q16 * 4)
                x = [(vis >= i).select(v[i], neg_inf) for i in range_constexpr(4)]
                m = _maxf_nn(_maxf_nn(x[0], x[1]), _maxf_nn(x[2], x[3]))
                return _maxf_nn(run, m)

            def finish(blk, run):
                sc, _ = _xlane_max4_pair(run, run)
                visible = blk < nb_row
                sc = (visible & (blk >= loc0)).select(f_local, sc)
                sc = (visible & (blk < i32_init)).select(f_init, sc)
                ok = (l16 < i32_qlen) & (q16 == 0) & (blk < blk1)
                buffer_ops.buffer_store(sc, sr, (r * i32_qlen + l16) * i32_S + blk, mask=ok)

            # two blocks per iteration; carried: pages of the two blocks, the two
            # pieces in flight (pieces 0, 1 of the iteration's first block)
            n_it = (blk1 - blk0 + 1) // 2
            p0 = page_of(blk0)
            p1 = page_of(blk0 + 1)
            assert RING <= PIECES  # the carried pieces belong to the iteration's first block
            ring_init = []
            for k in range_constexpr(RING):
                ring_init += load_piece(p0, k)
            init = [p0, p1] + ring_init
            for iv, st in range(
                fx.Index(0), fx.Index(n_it), fx.Index(1), init=[x.ir_value() for x in init]
            ):
                pg = [fx.Int32(st[0]), fx.Int32(st[1])]
                ring = [
                    [fx.Vector(st[2 + k * PIECE_LOADS + i]) for i in range_constexpr(PIECE_LOADS)]
                    for k in range_constexpr(RING)
                ]
                blk = blk0 + fx.Int32(iv) * 2
                nxt = [page_of(blk + 2), page_of(blk + 3)]
                run = [neg_inf, neg_inf]
                for k in range_constexpr(2 * PIECES):
                    kk = k + RING
                    if kk < 2 * PIECES:
                        ring.append(load_piece(pg[kk // PIECES], kk % PIECES))
                    else:
                        ring.append(load_piece(nxt[0], kk - 2 * PIECES))  # next iteration's
                    regs = ring.pop(0)
                    slot = k % RING
                    stage_piece(regs, slot)
                    b, p = k // PIECES, k % PIECES
                    for t in range_constexpr(2):
                        run[b] = mtile(slot, t, blk + b, 2 * p + t, run[b])
                    if p == PIECES - 1:
                        finish(blk + b, run[b])
                carried = [nxt[0], nxt[1]]
                for k in range_constexpr(RING):
                    carried += ring[k]
                res = yield [x.ir_value() for x in carried]

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nreq: fx.Int32,
        i32_qlen: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_total_q: fx.Int32,
        i64_kv_bytes: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_q, arg_kv, arg_score, arg_bt, arg_seq, i32_bt_stride, i32_S, i32_nreq,
            i32_qlen, i32_init, i32_local, i32_total_q, i64_kv_bytes,
        ).launch(grid=(WAVES // NW, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
