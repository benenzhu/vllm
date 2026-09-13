# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode index scorer for the fp8 (e4m3) index cache: the fp8 port of
``index_bf16.score_decode_v3``, a drop-in for AITER's
``pa_sparse_block_score_decode`` (uniform query length per request, ≤ 16 rows,
the same score buffer, causal masking and init / local sentinels).

Each wave owns a run of one request's blocks (wave-parallel work split: one
lane per request, wave prefix sums, a fixed budget of 1024 waves) and streams
them in 32-token pieces (4 KB): four 1 KB row-contiguous loads per piece (one
wave instruction covers eight 128 B token rows), written to the wave's own
padded LDS slot (144 B row stride: the 16 rows of an MFMA A operand hit 16
distinct bank groups) and read back as the fp8 MFMA A operand, one
``v_mfma_f32_16x16x128_f8f6f4`` per 16-token M-tile; no workgroup barrier, the
LDS region is private to the wave. Two pieces stay in flight across the loop
(loop-carried registers), the next two blocks' pages are loaded one iteration
ahead. The ≤ 16 query rows of the request are the MFMA B operand in 8 VGPRs.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from .utils import (
    _global_i32_ptr,
    _maxf_nn,
    _mfma_fp8_16x16x128,
    _pack8,
    _wave_prefix_sum_i32,
    _xlane_max4_pair,
)

NW = 4
# graph-constant wave budget (bf16 measured best: 4-5 blocks per wave, 800K, conc 1)
WAVES = 1024
BLK = 128
D = 128
ROW_BYTES = D
BLOCK_BYTES = BLK * ROW_BYTES  # 16 KB
PIECE_TOK = 32  # tokens per piece (2 M-tiles)
PIECES = BLK // PIECE_TOK
PIECE_LOADS = PIECE_TOK * ROW_BYTES // 1024  # 1 KB per wave instruction
RING = 2  # pieces in flight
LDS_PAD = 16
RS = ROW_BYTES + LDS_PAD
RS_T = RS // 16
SLOT_T = PIECE_TOK * RS_T  # 16 B tiles per piece slot
WAVE_T = RING * SLOT_T
LDS_BYTES = NW * WAVE_T * 16
MAX_REQS = 64  # one lane per request in the work split (the host launches per group)
K_CACHE_MOD = 2  # non-temporal K loads: the cache is streamed once (+10% for bf16)


def compile_score_decode():
    name = f"m3_index_score_decode_fp8_v3_w{WAVES}_ring{RING}"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # idx_q [total_q, 1, 128] e4m3
        arg_kv: fx.Int64,  # index cache [num_blocks, 128, 128] e4m3
        arg_score: fx.Int64,  # score [1, total_q, S] fp32
        arg_bt: fx.Int64,  # block table [num_reqs, stride] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] i32
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nreq: fx.Int32,
        i32_qlen: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_total_q: fx.Int32,
    ):
        k_cache_mod = K_CACHE_MOD
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        wid = pid * NW + wave
        seq = _global_i32_ptr(arg_seq)
        bt = _global_i32_ptr(arg_bt)

        # work split, one lane per request: request i gets floor(nb_i * (WAVES -
        # nreq) / total) waves, at least one, so the sum never exceeds WAVES
        # (rounding up per request could overflow the budget by nreq - 1 waves
        # and leave the last requests' tails unscored); its waves then split its
        # blocks evenly
        has = lane < i32_nreq
        L_l = fx.Int32(seq[has.select(lane, fx.Int32(0))])
        L_l = (has & (L_l > 0)).select(L_l, fx.Int32(0))
        nb_l = (L_l + (BLK - 1)) // BLK
        incl_nb = _wave_prefix_sum_i32(nb_l, lane)
        total_blocks = fx.Int32(fx.rocdl.readlane(T.i32, incl_nb, 63))
        total_blocks = (total_blocks > 0).select(total_blocks, fx.Int32(1))
        nw_l = (nb_l * (WAVES - i32_nreq)) // total_blocks
        nw_l = ((nb_l > 0) & (nw_l < 1)).select(fx.Int32(1), nw_l)
        incl_nw = _wave_prefix_sum_i32(nw_l, lane)
        off_l = incl_nw - nw_l
        owns = has & (wid >= off_l) & (wid < off_l + nw_l)
        own_mask = fx.Int64(fx.rocdl.ballot(T.i64, owns))
        r = fx.Int64(fx.math.cttz(own_mask)).to(fx.Int32)
        r = (own_mask != 0).select(r, fx.Int32(-1))
        r_ok = (own_mask != 0).select(r, fx.Int32(0))
        my_w = wid - fx.Int32(fx.rocdl.readlane(T.i32, off_l, r_ok))
        nb = fx.Int32(fx.rocdl.readlane(T.i32, nb_l, r_ok))
        nw = fx.Int32(fx.rocdl.readlane(T.i32, nw_l, r_ok))
        L = fx.Int32(fx.rocdl.readlane(T.i32, L_l, r_ok))
        if r >= 0:
            bpw = (nb + nw - 1) // nw
            blk0 = my_w * bpw
            blk1e = blk0 + bpw
            blk1 = (blk1e < nb).select(blk1e, nb)
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=fx.Int64(i32_total_q) * ROW_BYTES
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_score, num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4
            )
            # B operand (query rows l16): lane (l16, q16) holds dims [16*q16, +16) and
            # [64 + 16*q16, +16), the fp8 MFMA's K permutation, same as the A operand
            row = r * i32_qlen + l16
            qh = [
                fx.Vector(
                    buffer_ops.buffer_load(
                        qr,
                        (row * ROW_BYTES + h * 64 + q16 * 16) // 4,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                )
                for h in range_constexpr(2)
            ]
            qf = _pack8(qh[0], qh[1])
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
                return fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, fx.Int32(bt[bt_row + bi]))
                )

            # LDS: this wave's RING piece slots, 16 B tiles
            lds16 = fx.logical_divide(
                fx.make_view(
                    fx.recast_iter(fx.Int32, smem.a.ptr),
                    fx.make_layout(LDS_BYTES // 4, 1),
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

            # row-contiguous loads: instruction i of a piece covers rows 8i .. 8i+7,
            # lane (row 8i + lane//8, 16 B chunk lane%8); page and piece in soffset
            row_off = ((lane // 8) * ROW_BYTES + (lane % 8) * 16) // 4
            st_tiles = [
                (8 * i + lane // 8) * RS_T + lane % 8
                for i in range_constexpr(PIECE_LOADS)
            ]

            def load_piece(page, p):
                # one 16 KB buffer resource per block (64-bit base: the index cache may
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
                            soffset_bytes=p * (PIECE_TOK * ROW_BYTES) + i * 1024,
                        )
                    )
                    for i in range_constexpr(PIECE_LOADS)
                ]

            def stage_piece(regs, slot):
                for i in range_constexpr(PIECE_LOADS):
                    lds_store16(wave_t + slot * SLOT_T + st_tiles[i], regs[i])

            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            neg_inf = fx.Float32(float("-inf"))
            f_local = fx.Float32(1e29)
            f_init = fx.Float32(1e30)
            rd_base = l16 * RS_T

            def mtile(slot, mt_in_piece, blk, mt, run):
                base = wave_t + slot * SLOT_T + mt_in_piece * 16 * RS_T + rd_base
                a = _pack8(lds_load16(base + q16), lds_load16(base + 4 + q16))
                v = _mfma_fp8_16x16x128(a, qf, zero4)
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
                buffer_ops.buffer_store(
                    sc, sr, (r * i32_qlen + l16) * i32_S + blk, mask=ok
                )

            # two blocks per iteration; carried: pages of the two blocks, the two
            # pieces in flight (pieces 0, 1 of the iteration's first block)
            n_it = (blk1 - blk0 + 1) // 2
            p0 = page_of(blk0)
            p1 = page_of(blk0 + 1)
            assert (
                RING <= PIECES
            )  # the carried pieces belong to the iteration's first block
            ring_init = []
            for k in range_constexpr(RING):
                ring_init += load_piece(p0, k)
            init = [p0, p1] + ring_init
            for iv, st in range(
                fx.Index(0),
                fx.Index(n_it),
                fx.Index(1),
                init=[x.ir_value() for x in init],
            ):
                pg = [fx.Int32(st[0]), fx.Int32(st[1])]
                ring = [
                    [
                        fx.Vector(st[2 + k * PIECE_LOADS + i])
                        for i in range_constexpr(PIECE_LOADS)
                    ]
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
                        ring.append(
                            load_piece(nxt[0], kk - 2 * PIECES)
                        )  # next iteration's
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
                yield [x.ir_value() for x in carried]

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
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_score,
            arg_bt,
            arg_seq,
            i32_bt_stride,
            i32_S,
            i32_nreq,
            i32_qlen,
            i32_init,
            i32_local,
            i32_total_q,
        ).launch(grid=(WAVES // NW, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
