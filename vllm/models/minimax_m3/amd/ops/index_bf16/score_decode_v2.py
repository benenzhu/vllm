# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode index scorer for the bf16 index cache (gfx950, one local index head): the
score phase of ``minimax_m3_index_decode``; the fused Triton top-k / sparse-table
selector consumes the result unchanged.

``score[q, blk]`` = max over the visible tokens of block ``blk`` of ``q_row . k``
for the ``decode_query_len`` query rows of every request (row q of request r sits at
position ``seq_len - decode_query_len + q``); init blocks are forced to 1e30 and the
local block(s) to 1e29 on the visible blocks of a row, other blocks are -inf: the
Triton scorer's contract.

Pure K streaming. 4096 waves (1024 workgroups of 4, a graph-constant grid) split the
requests' blocks by real count: a wave owns a contiguous run of one request's blocks
(``bpw = cdiv(total_blocks, 4096)`` of them). It keeps the request's <= 16 query rows
(one MFMA N-tile; rows past decode_query_len are junk and never stored) as the B
operand in 16 VGPRs and streams K straight from global into A fragments: lane
(token, klane) reads 16 B at dims ku*32 + klane*8, so every instruction covers 64 B
contiguous per token row. Blocks go two per loop iteration; their 16 M-tile pairs
(8 KB each) are loaded two pairs ahead (64 VGPRs in flight) with non-temporal loads
(K is read once per layer), and the pages of the next two blocks are loaded one
iteration ahead (loop-carried).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from .utils import _global_i32_ptr, _maxf_nn, _wave_prefix_sum_i32, _xlane_max4_pair

NW = 4  # waves per workgroup
WAVES = 4096  # graph-constant wave budget
BLK = 128
D = 128
BLOCK_BYTES = BLK * D * 2
MT = BLK // 16  # token M-tiles per block
KU = D // 32
PAIRS = MT // 2  # M-tile pairs per block (8 KB each)
RING = 2  # pairs in flight
import os

_KNOBS = set(filter(None, os.environ.get("M3_IDX_KNOBS", "").split(",")))
K_CACHE_MOD = 0 if "nont" in _KNOBS else 2  # non-temporal K loads
MAX_REQS = 64  # one lane per request in the wave-parallel work split


def compile_score_decode():
    name = f"m3_index_score_decode_bf16_v2_w{WAVES}_ring{RING}" + "".join("_" + k for k in sorted(_KNOBS))

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # idx_q [total_q, 1, 128] bf16 (request-major, qlen rows each)
        arg_kv: fx.Int64,
        arg_score: fx.Int64,  # [1, total_q, S] fp32
        arg_bt: fx.Int64,  # [num_reqs, >= max_block] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] i32
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nreq: fx.Int32,
        i32_qlen: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_total_q: fx.Int32,
        i64_kv_bytes: fx.Int64,
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        wid = pid * NW + wave
        seq = _global_i32_ptr(arg_seq)
        bt = _global_i32_ptr(arg_bt)

        # work split, one lane per request: blocks per request, wave sum -> blocks per
        # wave, waves per request, prefix sum -> the request that owns this wave
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
            kr = buffer_ops.create_buffer_resource_from_addr(
                arg_kv, num_records_bytes=i64_kv_bytes
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_score, num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4
            )
            # query rows n = l16 (< qlen real): B fragments, lane (n, q16) at K-step ku
            # holds dims ku*32 + q16*8 .. +8
            row = r * i32_qlen + l16
            qf = [
                fx.Vector(
                    buffer_ops.buffer_load(
                        qr,
                        (row * (D * 2) + ku * 64 + q16 * 16) // 4,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                ).bitcast(fx.BFloat16)
                for ku in range_constexpr(KU)
            ]
            qpos = L - i32_qlen + l16  # query position of row l16
            kv_len = qpos + 1
            kv_len = (kv_len > 0).select(kv_len, fx.Int32(0))
            nb_row = (kv_len + (BLK - 1)) // BLK  # visible blocks of the row
            loc0 = nb_row - i32_local
            loc0 = (loc0 > 0).select(loc0, fx.Int32(0))
            bt_row = r * i32_bt_stride
            last_blk = nb - 1

            def page_of(blk):
                bi = (blk < nb).select(blk, last_blk)
                return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(bt[bt_row + bi])))

            # A fragment loads of M-tile pair p: rows (2p + t)*16 + l16, 16 B at
            # ku*64 + q16*16. One lane offset (dwords); the page and the M-tile go in
            # soffset (SGPR), ku*64 folds into the instruction offset field.
            a_off = (l16 * (D * 2) + q16 * 16) // 4

            def load_pair(page, p):
                return [
                    [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                kr,
                                a_off + ku * 16,
                                vec_width=4,
                                dtype=fx.Int32,
                                cache_modifier=K_CACHE_MOD,
                                soffset_bytes=page * BLOCK_BYTES + (2 * p + t) * 16 * (D * 2),
                            )
                        )
                        for ku in range_constexpr(KU)
                    ]
                    for t in range_constexpr(2)
                ]

            mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            neg_inf = fx.Float32(float("-inf"))
            f_local = fx.Float32(1e29)
            f_init = fx.Float32(1e30)

            def frag8(v8):
                t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                t.store(v8)
                return t

            def mtile(regs, blk, mt, run):
                acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                acc.store(zero4)
                for ku in range_constexpr(KU):
                    fx.gemm(mma, acc, frag8(regs[ku].bitcast(fx.BFloat16)), frag8(qf[ku]), acc)
                v = acc.load()
                vis = qpos - (blk * BLK + mt * 16 + q16 * 4)  # token i visible iff i <= vis
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

            # two blocks per iteration; their pages loaded one iteration ahead
            n_it = (blk1 - blk0 + 1) // 2
            p0 = page_of(blk0)
            p1 = page_of(blk0 + 1)
            for iv, st in range(
                fx.Index(0), fx.Index(n_it), fx.Index(1), init=[p0.ir_value(), p1.ir_value()]
            ):
                pg = [fx.Int32(st[0]), fx.Int32(st[1])]
                blk = blk0 + fx.Int32(iv) * 2
                nxt = [page_of(blk + 2), page_of(blk + 3)]
                # ring over the 8 pairs of the two blocks
                ring = [load_pair(pg[0], 0), load_pair(pg[0], 1)]
                run = [neg_inf, neg_inf]
                for k in range_constexpr(2 * PAIRS):
                    if k + RING < 2 * PAIRS:
                        kk = k + RING
                        ring.append(load_pair(pg[kk // PAIRS], kk % PAIRS))
                    regs = ring.pop(0)
                    b, p = k // PAIRS, k % PAIRS
                    for t in range_constexpr(2):
                        run[b] = mtile(regs[t], blk + b, 2 * p + t, run[b])
                    if p == PAIRS - 1:
                        finish(blk + b, run[b])
                res = yield [nxt[0].ir_value(), nxt[1].ir_value()]

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
