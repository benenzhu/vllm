# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k for uniform (decode) rows with four waves per row: the ``topk`` kernel
for few rows, where a single wave per row is bound by its own instruction
stream (25 chunks of 256 blocks at 650K context, a ballot and a shift per value
above the threshold on each: 14 us at 16 rows with the memory system idle).

One workgroup per row. Each wave scans a quarter of the row's chunks with the
seed / threshold list of ``topk`` (its first chunk seeds the list with 16
wave-argmax rounds, the later chunks only insert values above the threshold),
the four 16-entry lists meet in LDS, and wave 0 takes the 16 best of the 64
candidates with 16 wave-argmax rounds, writes the row's block ids and emits the
attend's page table exactly as ``topk`` does. Same output contract; among equal
scores the block order can differ from AITER's.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T

from .topk import (
    _ROW_SHR_1,
    BLK,
    CHUNK,
    INT32_MIN,
    PF_CHUNKS,
    PPB,
    TOPK,
    _dpp,
    _f32_to_ord,
    _imax,
    _popcount,
    _wave_max_i32,
)
from .utils import _global_i32_ptr

WPR = 4  # waves per row; the merge holds WPR * TOPK candidates, one per lane


def compile_topk_split():
    name = f"m3_index_topk_split_fp8_k{TOPK}_w{WPR}_c{CHUNK}_pf{PF_CHUNKS}_ppb{PPB}"

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, 2 * WPR * TOPK]  # the waves' lists: ords, then ids

    @flyc.kernel(name=name, known_block_size=[64 * WPR, 1, 1])
    def kernel(
        arg_score: fx.Int64,  # [1, total_q, S] fp32
        arg_out: fx.Int64,  # topk_idx [1, total_q, 16] i32 (row stride i32_out_stride)
        arg_bt: fx.Int64,  # the attend's (page-16) block table [num_reqs, stride] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs]
        arg_sbt: fx.Int64,  # sparse_bt [total_q * kvh, >= 16 * PPB] i32
        arg_sctx: fx.Int64,  # sparse_ctx [total_q * kvh] i32
        i64_seq_bytes: fx.Int64,
        i32_S: fx.Int32,
        i32_total_q: fx.Int32,
        i32_qlen: fx.Int32,  # query tokens per request
        i32_out_stride: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_sbt_stride: fx.Int32,
        i32_kvh: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        cand = smem.cand.ptr
        tx, n = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        sres = buffer_ops.create_buffer_resource_from_addr(
            arg_seq, num_records_bytes=i64_seq_bytes
        )
        req = n // i32_qlen
        seq = fx.Int32(buffer_ops.buffer_load(sres, req, vec_width=1, dtype=fx.Int32))
        causal = seq - i32_qlen + (n - req * i32_qlen) + 1
        causal = (causal > 0).select(causal, fx.Int32(0))
        vb = (causal + (BLK - 1)) // BLK  # visible blocks
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_score, num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4
        )
        row_base = n * i32_S
        int_min = fx.Int32(INT32_MIN)
        f_nan = fx.Float32(-1e30)

        # this wave's chunks [c_lo, c_hi): a wave past the row's chunks scans
        # nothing (its seed chunk is masked off entirely)
        n_chunks = (vb + (CHUNK - 1)) // CHUNK
        per = (n_chunks + (WPR - 1)) // WPR
        c_lo = wave * per
        c_hi = c_lo + per
        c_hi = (c_hi < n_chunks).select(c_hi, n_chunks)

        def load_raw(c):
            return fx.Vector(
                buffer_ops.buffer_load(
                    sr,
                    row_base + c * CHUNK + lane * 4,
                    vec_width=4,
                    dtype=fx.Float32,
                )
            )

        def unpack(v4, c, full):
            """ord keys of the chunk's 4 values per lane. ``full``: the wave's
            first chunk and the row's last one carry the visible-block bound and
            the NaN map; the chunks between are all visible finite scores (the
            scorer masks with -inf, never NaN)."""
            col0 = c * CHUNK + lane * 4
            ords, cols = [], []
            for j in range_constexpr(4):
                col = col0 + j
                v = v4[j]
                if const_expr(full):
                    bits = v.bitcast(fx.Int32)
                    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
                    v = is_nan.select(f_nan, v)
                    ords.append((col < vb).select(_f32_to_ord(v), int_min))
                else:
                    ords.append(_f32_to_ord(v))
                cols.append(col)
            return ords, cols

        # seed: 16 wave-argmax rounds over the wave's first chunk, the next PF
        # chunks in flight. Ties go to the higher block id (AITER's key order).
        ords, cols = unpack(load_raw(c_lo), c_lo, True)
        ring = [load_raw(c_lo + (1 + i)) for i in range(PF_CHUNKS)]
        list_ord = int_min
        list_idx = fx.Int32(-1)
        for r in range_constexpr(TOPK):
            cur = _imax(_imax(ords[0], ords[1]), _imax(ords[2], ords[3]))
            m = _wave_max_i32(cur)
            win = fx.Int64(_rocdl.ballot(T.i64, cur == m))
            lane_w = fx.Int32(63) - fx.Int64(fx.math.ctlz(win)).to(fx.Int32)
            idx_loc = cols[0]
            for j in range_constexpr(1, 4):
                idx_loc = (ords[j] == m).select(cols[j], idx_loc)
            idx_w = fx.Int32(_rocdl.readlane(T.i32, idx_loc, lane_w))
            entry = (m == int_min).select(fx.Int32(-1), idx_w)
            is_r = lane == r
            list_ord = is_r.select(m, list_ord)
            list_idx = is_r.select(entry, list_idx)
            is_w = lane == lane_w
            taken = fx.Int32(0)
            for j in range_constexpr(3, -1, -1):
                hit = is_w & (ords[j] == m) & (taken == 0)
                ords[j] = hit.select(int_min, ords[j])
                taken = hit.select(fx.Int32(1), taken)
        thr = fx.Int32(_rocdl.readlane(T.i32, list_ord, 15))

        def scan(civ, st, full):
            """one chunk: consume ring[0], refill the ring, insert the hits"""
            list_ord = fx.Int32(st[0])
            list_idx = fx.Int32(st[1])
            thr = fx.Int32(st[2])
            ring = [fx.Vector(st[3 + i]) for i in range(PF_CHUNKS)]
            cur = ring[0]
            ring = ring[1:] + [load_raw(fx.Int32(civ) + PF_CHUNKS)]
            ords, cols = unpack(cur, fx.Int32(civ), full)
            for j in range_constexpr(4):
                mask = fx.Int64(_rocdl.ballot(T.i64, ords[j] > thr))
                n_hit = _popcount(mask)
                for kiv, st2 in range(
                    fx.Index(0),
                    fx.Index(n_hit),
                    fx.Index(1),
                    init=[x.ir_value() for x in (list_ord, list_idx, thr, mask)],
                ):
                    list_ord = fx.Int32(st2[0])
                    list_idx = fx.Int32(st2[1])
                    thr = fx.Int32(st2[2])
                    mask = fx.Int64(st2[3])
                    lane_w = fx.Int64(fx.math.cttz(mask)).to(fx.Int32)
                    mask = mask & (mask - 1)
                    oj = fx.Int32(_rocdl.readlane(T.i32, ords[j], lane_w))
                    ij = fx.Int32(_rocdl.readlane(T.i32, cols[j], lane_w))
                    ins = oj > thr
                    gt = fx.Int64(_rocdl.ballot(T.i64, list_ord > oj)) & 0xFFFF
                    cnt = _popcount(gt)
                    p = ins.select(cnt, fx.Int32(TOPK))
                    sh_o = _dpp(list_ord, _ROW_SHR_1)
                    sh_i = _dpp(list_idx, _ROW_SHR_1)
                    below = lane > p
                    list_ord = (lane == p).select(oj, below.select(sh_o, list_ord))
                    list_idx = (lane == p).select(ij, below.select(sh_i, list_idx))
                    thr = fx.Int32(_rocdl.readlane(T.i32, list_ord, 15))
                    res2 = yield [x.ir_value() for x in (list_ord, list_idx, thr, mask)]
                list_ord = fx.Int32(res2[0])
                list_idx = fx.Int32(res2[1])
                thr = fx.Int32(res2[2])
            return [x.ir_value() for x in (list_ord, list_idx, thr, *ring)]

        # the wave's remaining chunks: the cheap unpack up to the row's last
        # chunk, which carries the visible-block bound (both loops are empty for
        # a wave with one chunk or none)
        lo1 = c_lo + 1
        m_hi = n_chunks - 1
        m_hi = (c_hi < m_hi).select(c_hi, m_hi)
        m_hi = (m_hi > lo1).select(m_hi, lo1)
        for civ, st in range(
            fx.Index(lo1),
            fx.Index(m_hi),
            fx.Index(1),
            init=[x.ir_value() for x in (list_ord, list_idx, thr, *ring)],
        ):
            res = yield scan(civ, st, False)
        for civ, st in range(
            fx.Index(m_hi),
            fx.Index(c_hi),
            fx.Index(1),
            init=list(res),
        ):
            res = yield scan(civ, st, True)
        list_ord = fx.Int32(res[0])
        list_idx = fx.Int32(res[1])

        # merge: the lists in LDS, the later waves (higher block ids) first so
        # that among equal scores the lowest lane holds the higher block id
        slot = (WPR - 1 - wave) * TOPK + lane
        if lane < TOPK:
            cand[slot] = list_ord
            cand[WPR * TOPK + slot] = list_idx
        fx.gpu.barrier()
        if wave == 0:
            o = fx.Int32(cand[lane])
            i = fx.Int32(cand[WPR * TOPK + lane])
            sel = fx.Int32(-1)
            for r in range_constexpr(TOPK):
                m = _wave_max_i32(o)
                win = fx.Int64(_rocdl.ballot(T.i64, o == m))
                lane_w = fx.Int64(fx.math.cttz(win)).to(fx.Int32)
                idx_w = fx.Int32(_rocdl.readlane(T.i32, i, lane_w))
                entry = (m == int_min).select(fx.Int32(-1), idx_w)
                sel = (lane == r).select(entry, sel)
                o = (lane == lane_w).select(int_min, o)
            out = _global_i32_ptr(arg_out)
            if lane < TOPK:
                out[n * i32_out_stride + lane] = sel

            # the attend's page table for this row (lanes 0..15 hold the entries)
            self_blk = (causal > 0).select((causal - 1) // BLK, fx.Int32(0))
            blk = sel
            valid = (lane < TOPK) & (causal > 0) & (blk >= 0) & (blk <= self_blk)
            is_full = valid & (blk < self_blk)
            valid_mask = fx.Int64(_rocdl.ballot(T.i64, valid))
            full_mask = fx.Int64(_rocdl.ballot(T.i64, is_full))
            n_valid = _popcount(valid_mask)
            n_full = _popcount(full_mask)
            # full blocks packed in score order, the row's own block right after
            preceding = full_mask & ((fx.Int64(1) << fx.Int64(lane)) - 1)
            pslot = is_full.select(_popcount(preceding), n_full)
            page = fx.Int32(
                _global_i32_ptr(arg_bt)[
                    req * i32_bt_stride + valid.select(blk, fx.Int32(0))
                ]
            )
            has_tail = n_valid > n_full
            tail_tokens = n_full * BLK + causal - self_blk * BLK
            full_tokens = (n_valid * BLK < causal).select(n_valid * BLK, causal)
            ctx_tokens = has_tail.select(tail_tokens, full_tokens)
            # lanes past the valid entries zero their own slot (slots n_valid .. 15)
            is_zero = (lane >= n_valid) & (lane < TOPK)
            wr = valid | is_zero
            slot_w = valid.select(pslot, lane)
            sbt = buffer_ops.create_buffer_resource_from_addr(
                arg_sbt,
                num_records_bytes=fx.Int64(i32_total_q)
                * fx.Int64(i32_kvh)
                * fx.Int64(i32_sbt_stride)
                * 4,
            )
            sctx = _global_i32_ptr(arg_sctx)
            for h in range(fx.Int32(0), i32_kvh, fx.Int32(1)):
                hh = fx.Int32(h)
                base = page * (PPB * i32_kvh) + hh
                dst = (n * i32_kvh + hh) * i32_sbt_stride + slot_w * PPB
                for half in range_constexpr(2):
                    vals = [
                        valid.select(base + (half * 4 + j) * i32_kvh, fx.Int32(0))
                        for j in range(4)
                    ]
                    buffer_ops.buffer_store(
                        fx.Vector.from_elements(vals, fx.Int32),
                        sbt,
                        dst + half * 4,
                        mask=wr,
                    )
                if lane == 0:
                    sctx[n * i32_kvh + hh] = ctx_tokens

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        arg_out: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_sbt: fx.Int64,
        arg_sctx: fx.Int64,
        i64_seq_bytes: fx.Int64,
        i32_S: fx.Int32,
        i32_total_q: fx.Int32,
        i32_qlen: fx.Int32,
        i32_out_stride: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_sbt_stride: fx.Int32,
        i32_kvh: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            arg_out,
            arg_bt,
            arg_seq,
            arg_sbt,
            arg_sctx,
            i64_seq_bytes,
            i32_S,
            i32_total_q,
            i32_qlen,
            i32_out_stride,
            i32_bt_stride,
            i32_sbt_stride,
            i32_kvh,
        ).launch(
            grid=(fx.Int64(i32_total_q), 1, 1), block=(64 * WPR, 1, 1), stream=stream
        )

    launch.kernel_name = name
    return launch
