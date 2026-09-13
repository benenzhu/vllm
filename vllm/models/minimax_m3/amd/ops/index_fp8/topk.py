# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k over the index scores with the attend's sparse page table emitted
alongside: a drop-in for AITER's ``pa_sparse_block_topk`` (one local head,
topk 16, 8 pages per 128-token block).

For every score row: the 16 highest-scoring blocks among its causal blocks
(``blk < cdiv(causal_len, 128)``), block ids in descending score order, ``-1``
where fewer blocks are visible. The scorer already wrote the init / local
sentinels (1e30 / 1e29) into the row, so they rank first by value; NaN ranks
last. Rows are either ragged (prefill: ``row_req_id`` / ``kv_lens`` per row) or
uniform (decode: request ``n // query_len``, causal length from ``seq_lens``).

One wave per row, four rows (waves) per workgroup. The wave keeps the sorted
16-entry (score, block) list in lanes 0..15 and the 16th score, the threshold,
in an SGPR. Scores are compared as order-preserving int32. Chunk 0 (blocks
0..255, 4 per lane) seeds the list with 16 wave-argmax rounds; every later
256-block chunk is loaded 16 B per lane, and only the values above the
threshold (about 16 * ln(blocks / 16) per row over the whole scan) are inserted,
one at a time, with a 16-lane shift. Memory-bound with thousands of rows (one
read of the score row); with the few rows of a decode step one wave per row is
bound by its own instruction stream, so uniform rows go to ``topk_split``
(four waves per row) instead.

The table: the selected blocks' pages packed towards slot 0 in score order,
full blocks first and the row's own (partial) block last, zeros after, one row
per (token, kv head) with the page ids folded head-minor, and the token count
those pages hold -- AITER's ``emit_sparse_block_table_row`` contract, which the
AITER sparse-PA attend reads.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T

from .utils import _global_i32_ptr, _permlane16_swap, _permlane32_swap

NW = 4  # rows (waves) per workgroup
TOPK = 16
BLK = 128
PPB = 8  # pages per block (ASM page 16 tokens)
CHUNK = 256  # blocks per chunk: 4 per lane, one dwordx4
# chunks loaded ahead of the one being scanned (the scan of a row is a chain of
# dependent 1 KB loads otherwise: ~20 round trips at 650K context)
PF_CHUNKS = 2
INT32_MIN = -(1 << 31)
# DPP controls (16-lane rows): row_ror:n = 0x120 + n, row_shr:n = 0x110 + n
_ROW_ROR = (0x121, 0x122, 0x124, 0x128)
_ROW_SHR_1 = 0x111


def _dpp(v, ctrl):
    """``update_dpp`` with the value itself as the fallback for lanes without a
    source (row_shr lane 0)."""
    raw = fx.as_ir_value(v)
    return fx.Int32(_rocdl.update_dpp(T.i32, raw, raw, ctrl, 0xF, 0xF, False))


def _imax(a, b):
    return (a > b).select(a, b)


def _wave_max_i32(v):
    """Max over the 64 lanes, in every lane: 4 row rotations, then the two
    permlane swaps of ``_xlane_max4_pair``."""
    for ctrl in _ROW_ROR:
        v = _imax(v, _dpp(v, ctrl))
    a, b = _permlane32_swap(v, v)
    v = _imax(a, b)
    a, b = _permlane16_swap(v, v)
    return _imax(a, b)


def _f32_to_ord(v):
    """Order-preserving float32 -> int32 (dcp_topk_merge's map); NaN handled
    before the call."""
    bits = v.bitcast(fx.Int32)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def _popcount(mask64):
    return fx.Int64(fx.math.ctpop(mask64)).to(fx.Int32)


def compile_topk():
    name = f"m3_index_topk_fp8_k{TOPK}_nw{NW}_c{CHUNK}_pf{PF_CHUNKS}_ppb{PPB}"

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_score: fx.Int64,  # [1, total_q, S] fp32
        arg_out: fx.Int64,  # topk_idx [1, total_q, 16] i32 (row stride i32_out_stride)
        arg_bt: fx.Int64,  # the attend's (page-16) block table [num_reqs, stride] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] (uniform rows)
        arg_rid: fx.Int64,  # row_req_id [total_q] (ragged rows)
        arg_kvl: fx.Int64,  # kv_lens [total_q] (ragged rows)
        arg_sbt: fx.Int64,  # sparse_bt [total_q * kvh, >= 16 * PPB] i32
        arg_sctx: fx.Int64,  # sparse_ctx [total_q * kvh] i32
        i64_seq_bytes: fx.Int64,
        i64_rows_bytes: fx.Int64,  # bytes of row_req_id / kv_lens; 0 with uniform rows
        i32_S: fx.Int32,
        i32_total_q: fx.Int32,
        i32_qlen: fx.Int32,  # query tokens per request; 0 = ragged rows
        i32_out_stride: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_sbt_stride: fx.Int32,
        i32_kvh: fx.Int32,
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        n = pid * NW + wave
        if n < i32_total_q:
            # the row's request and causal length, both shapes computed (the loads
            # of the shape not in use are out of their zero-length buffers: 0)
            ragged = i32_qlen == 0
            qlen1 = ragged.select(fx.Int32(1), i32_qlen)
            rr = buffer_ops.create_buffer_resource_from_addr(
                arg_rid, num_records_bytes=i64_rows_bytes
            )
            kr = buffer_ops.create_buffer_resource_from_addr(
                arg_kvl, num_records_bytes=i64_rows_bytes
            )
            sres = buffer_ops.create_buffer_resource_from_addr(
                arg_seq, num_records_bytes=i64_seq_bytes
            )
            req_r = fx.Int32(buffer_ops.buffer_load(rr, n, vec_width=1, dtype=fx.Int32))
            causal_r = fx.Int32(
                buffer_ops.buffer_load(kr, n, vec_width=1, dtype=fx.Int32)
            )
            req_u = n // qlen1
            seq_u = fx.Int32(
                buffer_ops.buffer_load(sres, req_u, vec_width=1, dtype=fx.Int32)
            )
            causal_u = seq_u - qlen1 + (n - req_u * qlen1) + 1
            req = ragged.select(req_r, req_u)
            causal = ragged.select(causal_r, causal_u)
            causal = (causal > 0).select(causal, fx.Int32(0))
            vb = (causal + (BLK - 1)) // BLK  # visible blocks
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_score, num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4
            )
            row_base = n * i32_S
            int_min = fx.Int32(INT32_MIN)
            f_nan = fx.Float32(-1e30)

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
                """ord keys of the chunk's 4 values per lane. ``full``: the first and
                the last chunk carry the visible-block bound and the NaN map; the
                chunks between are all visible finite scores (the scorer masks
                with -inf, never NaN)."""
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

            # seed: 16 wave-argmax rounds over chunk 0, chunks 1..PF in flight.
            # Ties go to the higher block id (AITER's key order).
            ords, cols = unpack(load_raw(fx.Int32(0)), fx.Int32(0), True)
            ring = [load_raw(fx.Int32(1 + i)) for i in range(PF_CHUNKS)]
            list_ord = int_min
            list_idx = fx.Int32(-1)
            for r in range_constexpr(TOPK):
                cur = _imax(_imax(ords[0], ords[1]), _imax(ords[2], ords[3]))
                m = _wave_max_i32(cur)
                win = fx.Int64(_rocdl.ballot(T.i64, cur == m))
                lane_w = fx.Int32(63) - fx.Int64(fx.math.ctlz(win)).to(fx.Int32)
                # the winner lane's last value equal to m (highest block id)
                idx_loc = cols[0]
                for j in range_constexpr(1, 4):
                    idx_loc = (ords[j] == m).select(cols[j], idx_loc)
                idx_w = fx.Int32(_rocdl.readlane(T.i32, idx_loc, lane_w))
                entry = (m == int_min).select(fx.Int32(-1), idx_w)
                is_r = lane == r
                list_ord = is_r.select(m, list_ord)
                list_idx = is_r.select(entry, list_idx)
                # remove it from the winner lane
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
                # loads past the row's chunks land in the masked tail / the next row
                ring = ring[1:] + [load_raw(fx.Int32(civ) + PF_CHUNKS)]
                ords, cols = unpack(cur, fx.Int32(civ), full)
                # candidates above the threshold, one value at a time (a hit lane
                # rarely has more than one): insert below the entries strictly
                # greater (equal ones fall under the newer, higher block id), shift
                # the rest down one lane, re-read the threshold
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
                        ins = oj > thr  # the threshold may have risen since the ballot
                        gt = fx.Int64(_rocdl.ballot(T.i64, list_ord > oj)) & 0xFFFF
                        cnt = _popcount(gt)
                        p = ins.select(cnt, fx.Int32(TOPK))
                        sh_o = _dpp(list_ord, _ROW_SHR_1)
                        sh_i = _dpp(list_idx, _ROW_SHR_1)
                        below = lane > p
                        list_ord = (lane == p).select(oj, below.select(sh_o, list_ord))
                        list_idx = (lane == p).select(ij, below.select(sh_i, list_idx))
                        thr = fx.Int32(_rocdl.readlane(T.i32, list_ord, 15))
                        res2 = yield [
                            x.ir_value() for x in (list_ord, list_idx, thr, mask)
                        ]
                    list_ord = fx.Int32(res2[0])
                    list_idx = fx.Int32(res2[1])
                    thr = fx.Int32(res2[2])
                return [x.ir_value() for x in (list_ord, list_idx, thr, *ring)]

            # remaining chunks: the middle ones with the cheap unpack, the last one
            # (it holds the visible-block bound) with the full one
            n_chunks = (vb + (CHUNK - 1)) // CHUNK
            n_last = (n_chunks > 1).select(n_chunks - 1, fx.Int32(1))
            for civ, st in range(
                fx.Index(1),
                fx.Index(n_last),
                fx.Index(1),
                init=[x.ir_value() for x in (list_ord, list_idx, thr, *ring)],
            ):
                res = yield scan(civ, st, False)
            for civ, st in range(
                fx.Index(n_last),
                fx.Index(n_chunks),
                fx.Index(1),
                init=list(res),
            ):
                res = yield scan(civ, st, True)
            list_idx = fx.Int32(res[1])
            out = _global_i32_ptr(arg_out)
            if lane < TOPK:
                out[n * i32_out_stride + lane] = list_idx

            # the attend's page table for this row: lanes 0..15 hold the entries
            # (a rejected insertion parks its value in lane 16: not an entry)
            self_blk = (causal > 0).select((causal - 1) // BLK, fx.Int32(0))
            blk = list_idx
            valid = (lane < TOPK) & (causal > 0) & (blk >= 0) & (blk <= self_blk)
            is_full = valid & (blk < self_blk)
            valid_mask = fx.Int64(_rocdl.ballot(T.i64, valid))
            full_mask = fx.Int64(_rocdl.ballot(T.i64, is_full))
            n_valid = _popcount(valid_mask)
            n_full = _popcount(full_mask)
            # full blocks packed in score order, the row's own block right after them
            preceding = full_mask & ((fx.Int64(1) << fx.Int64(lane)) - 1)
            slot = is_full.select(_popcount(preceding), n_full)
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
            slot_w = valid.select(slot, lane)
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
        arg_rid: fx.Int64,
        arg_kvl: fx.Int64,
        arg_sbt: fx.Int64,
        arg_sctx: fx.Int64,
        i64_seq_bytes: fx.Int64,
        i64_rows_bytes: fx.Int64,
        i32_S: fx.Int32,
        i32_total_q: fx.Int32,
        i32_qlen: fx.Int32,
        i32_out_stride: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_sbt_stride: fx.Int32,
        i32_kvh: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score,
            arg_out,
            arg_bt,
            arg_seq,
            arg_rid,
            arg_kvl,
            arg_sbt,
            arg_sctx,
            i64_seq_bytes,
            i64_rows_bytes,
            i32_S,
            i32_total_q,
            i32_qlen,
            i32_out_stride,
            i32_bt_stride,
            i32_sbt_stride,
            i32_kvh,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
