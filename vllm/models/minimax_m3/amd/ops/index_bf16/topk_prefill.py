# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill top-k over the index scores (one local head, topk 16).

For every query row: the 16 highest-scoring blocks among its causal blocks
(``blk < (prefix + row + 128) // 128``), block ids in descending score order,
``-1`` where fewer blocks are visible; same contract as the Triton
``_topk_index_kernel`` (init blocks forced with 1e30, the local block with 1e29,
NaN scores as -1e30).

One wave per row, four rows (waves) per workgroup. The wave keeps the sorted
16-entry (score, block) list in lanes 0..15 and the 16th score, the threshold,
in an SGPR. Scores are compared as order-preserving int32 (-inf and NaN at the
bottom). Chunk 0 (blocks 0..255, 4 per lane) seeds the list with 16 wave-argmax
rounds; every later 256-block chunk is loaded 16 B per lane, and only the
values above the threshold (about 16 * ln(blocks / 16) per row over the whole
scan) are inserted, one at a time, with a 16-lane shift. Equal scores keep the
lower block id first. Memory-bound: one read of the score row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr import rocdl as _rocdl
from flydsl.expr.typing import T

from .utils import _global_i32_ptr, _permlane16_swap, _permlane32_swap

NW = 4  # rows (waves) per workgroup
TOPK = 16
BLK = 128
CHUNK = 256  # blocks per chunk: 4 per lane, one dwordx4
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


def compile_topk_prefill():
    name = f"m3_index_topk_prefill_k{TOPK}_nw{NW}_c{CHUNK}"

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_score: fx.Int64,  # [1, total_q, S] fp32
        arg_out: fx.Int64,  # [1, total_q, 16] i32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1]
        arg_prefix: fx.Int64,  # prefix_lens [batch]
        i32_S: fx.Int32,
        i32_qbs: fx.Int32,  # row groups (of NW) per request
        i32_total_q: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        qb = pid % i32_qbs
        b = pid // i32_qbs
        cu = _global_i32_ptr(arg_cu)
        seq_start = fx.Int32(cu[b])
        q_len = fx.Int32(cu[b + 1]) - seq_start
        prefix = fx.Int32(_global_i32_ptr(arg_prefix)[b])
        r_local = qb * NW + wave
        if r_local < q_len:
            row = seq_start + r_local
            vb = (prefix + r_local + BLK) // BLK  # visible blocks
            loc0 = vb - i32_local
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_score, num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4
            )
            row_base = row * i32_S
            int_min = fx.Int32(INT32_MIN)
            f_nan = fx.Float32(-1e30)
            f_local = fx.Float32(1e29)
            f_init = fx.Float32(1e30)

            def load_chunk(c):
                col0 = c * CHUNK + lane * 4
                v4 = fx.Vector(
                    buffer_ops.buffer_load(
                        sr, row_base + col0, vec_width=4, dtype=fx.Float32
                    )
                )
                ords, cols = [], []
                for j in range_constexpr(4):
                    col = col0 + j
                    v = v4[j]
                    bits = v.bitcast(fx.Int32)
                    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
                    v = is_nan.select(f_nan, v)
                    v = (col >= loc0).select(f_local, v)
                    v = (col < i32_init).select(f_init, v)
                    ords.append((col < vb).select(_f32_to_ord(v), int_min))
                    cols.append(col)
                return ords, cols

            # seed: 16 wave-argmax rounds over chunk 0
            ords, cols = load_chunk(fx.Int32(0))
            list_ord = int_min
            list_idx = fx.Int32(-1)
            for r in range_constexpr(TOPK):
                cur = _imax(_imax(ords[0], ords[1]), _imax(ords[2], ords[3]))
                m = _wave_max_i32(cur)
                win = fx.Int64(_rocdl.ballot(T.i64, cur == m))
                lane_w = fx.Int64(fx.math.cttz(win)).to(fx.Int32)
                # the winner lane's first value equal to m (lowest block id)
                idx_loc = cols[3]
                for j in range_constexpr(2, -1, -1):
                    idx_loc = (ords[j] == m).select(cols[j], idx_loc)
                idx_w = fx.Int32(_rocdl.readlane(T.i32, idx_loc, lane_w))
                entry = (m == int_min).select(fx.Int32(-1), idx_w)
                is_r = lane == r
                list_ord = is_r.select(m, list_ord)
                list_idx = is_r.select(entry, list_idx)
                # remove it from the winner lane
                is_w = lane == lane_w
                taken = fx.Int32(0)
                for j in range_constexpr(4):
                    hit = is_w & (ords[j] == m) & (taken == 0)
                    ords[j] = hit.select(int_min, ords[j])
                    taken = hit.select(fx.Int32(1), taken)
            thr = fx.Int32(_rocdl.readlane(T.i32, list_ord, 15))

            # remaining chunks: insert the values above the threshold
            n_chunks = (vb + (CHUNK - 1)) // CHUNK
            for civ, st in range(
                fx.Index(1),
                fx.Index(n_chunks),
                fx.Index(1),
                init=[x.ir_value() for x in (list_ord, list_idx, thr)],
            ):
                list_ord = fx.Int32(st[0])
                list_idx = fx.Int32(st[1])
                thr = fx.Int32(st[2])
                ords, cols = load_chunk(fx.Int32(civ))
                hit = (ords[0] > thr) | (ords[1] > thr) | (ords[2] > thr) | (ords[3] > thr)
                mask = fx.Int64(_rocdl.ballot(T.i64, hit))
                n_hit = fx.Int64(fx.math.ctpop(mask)).to(fx.Int32)
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
                    for j in range_constexpr(4):
                        oj = fx.Int32(_rocdl.readlane(T.i32, ords[j], lane_w))
                        ij = fx.Int32(_rocdl.readlane(T.i32, cols[j], lane_w))
                        ins = oj > thr
                        ge = fx.Int64(_rocdl.ballot(T.i64, list_ord >= oj)) & 0xFFFF
                        cnt = fx.Int64(fx.math.ctpop(ge)).to(fx.Int32)
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
                res = yield [x.ir_value() for x in (list_ord, list_idx, thr)]
            list_idx = fx.Int32(res[1])
            out = _global_i32_ptr(arg_out)
            if lane < TOPK:
                out[row * TOPK + lane] = list_idx

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        arg_out: fx.Int64,
        arg_cu: fx.Int64,
        arg_prefix: fx.Int64,
        i32_S: fx.Int32,
        i32_qbs: fx.Int32,
        i32_total_q: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score, arg_out, arg_cu, arg_prefix, i32_S, i32_qbs, i32_total_q,
            i32_init, i32_local,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
