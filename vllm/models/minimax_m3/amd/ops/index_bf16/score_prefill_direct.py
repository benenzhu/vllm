# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill index scorer, direct variant: no LDS, no barriers.

Same contract, query tiling (4 waves x 128 rows = 512-row tile x ctx segment) and
per-wave math as ``score_prefill`` (Q^T in VGPRs as the MFMA B operand, C^T = K . Q^T,
in-lane + 4-lane max per block). The K block is not staged through LDS: every wave
streams its A fragments straight from global, lane (token, klane) reading 16 B at dims
ku*32 + klane*8 so each instruction covers 64 B contiguous per token row; the four
waves of a workgroup read the same block within a few microseconds of each other and
share it in L1/L2. A block's 8 token M-tiles go in 4 pairs (8 KB per pair), two pairs
in flight (64 VGPRs); four blocks per loop iteration, whose pages were loaded one
iteration ahead (loop-carried); scores are kept four blocks per row and stored 16 B
per row (single dwords for a segment's tail).

Measured against the LDS-staged kernel: the per-block workgroup barrier cost 20-30%
of the block time (all four waves resynchronise and drain every block); this variant
has none.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .utils import _global_i32_ptr, _maxf_nn, _xlane_max4_pair

NW = 4
ROWS_PER_WAVE = 128
TILE_Q = NW * ROWS_PER_WAVE
BLK = 128
D = 128
BLOCK_BYTES = BLK * D * 2
GROUP = 4  # blocks per iteration / score store
NI = ROWS_PER_WAVE // 16
MT = BLK // 16
KU = D // 32
PAIRS = MT // 2
RING = 2  # M-tile pairs in flight
_KNOBS = set(filter(None, os.environ.get("M3_IDX_KNOBS", "").split(",")))
K_CACHE_MOD = 2 if "nt" in _KNOBS else 0
NO_STORE = "no_store" in _KNOBS


def compile_score_prefill_direct():
    name = f"m3_index_score_prefill_bf16_direct_tq{TILE_Q}_ring{RING}" + "".join(
        "_" + k for k in sorted(_KNOBS)
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        arg_prefix: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i64_kv_bytes: fx.Int64,
        i32_nb: fx.Int32,
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        q_tile = pid % i32_qt
        seg = (pid // i32_qt) % i32_nseg
        b = pid // (i32_qt * i32_nseg)
        cu = _global_i32_ptr(arg_cu)
        seq_start = fx.Int32(cu[b])
        q_len = fx.Int32(cu[b + 1]) - seq_start
        seq_len = fx.Int32(_global_i32_ptr(arg_seq)[b])
        prefix = fx.Int32(_global_i32_ptr(arg_prefix)[b])
        bt = _global_i32_ptr(arg_bt)
        row0 = q_tile * TILE_Q
        if row0 < q_len:
            hi_pos = prefix + row0 + TILE_Q
            hi = (hi_pos < seq_len).select(hi_pos, seq_len)
            nblk = (hi + (BLK - 1)) // BLK
            seg_len = ((nblk + i32_nseg - 1) // i32_nseg + (GROUP - 1)) & ~(GROUP - 1)
            blk0 = seg * seg_len
            blk1e = blk0 + seg_len
            blk1 = (blk1e < nblk).select(blk1e, nblk)
            if blk0 < blk1:
                qr = buffer_ops.create_buffer_resource_from_addr(
                    arg_q, num_records_bytes=fx.Int64(i32_total_q) * (D * 2)
                )
                kr = buffer_ops.create_buffer_resource_from_addr(
                    arg_kv, num_records_bytes=i64_kv_bytes
                )
                sr = buffer_ops.create_buffer_resource_from_addr(
                    arg_score,
                    num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4,
                )
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r for r in rows]
                # B fragments: lane (row, q16) at K-step ku holds dims ku*32 + q16*8 .. +8
                qf = [
                    [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                qr,
                                ((seq_start + rows[ni]) * (D * 2) + ku * 64 + q16 * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        ).bitcast(fx.BFloat16)
                        for ku in range_constexpr(KU)
                    ]
                    for ni in range_constexpr(NI)
                ]
                bt_row = b * i32_bt_stride
                last_blk = nblk - 1

                def page_of(blk):
                    bi = (blk < nblk).select(blk, last_blk)
                    return fx.Int32(bt[bt_row + bi])

                a_off = l16 * (D * 2) + q16 * 16

                def load_pair(page, p):
                    base = page * (BLOCK_BYTES // 4)
                    return [
                        [
                            fx.Vector(
                                buffer_ops.buffer_load(
                                    kr,
                                    base + (a_off + (2 * p + t) * 16 * (D * 2) + ku * 64) // 4,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    cache_modifier=K_CACHE_MOD,
                                )
                            )
                            for ku in range_constexpr(KU)
                        ]
                        for t in range_constexpr(2)
                    ]

                mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))

                def frag8(v8):
                    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                    t.store(v8)
                    return t

                def mtile(regs, blk, mt, run):
                    a = [frag8(regs[ku].bitcast(fx.BFloat16)) for ku in range_constexpr(KU)]
                    acc = [
                        fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(NI)
                    ]
                    for ni in range_constexpr(NI):
                        acc[ni].store(zero4)
                    for ku in range_constexpr(KU):
                        for ni in range_constexpr(NI):
                            fx.gemm(mma, acc[ni], a[ku], frag8(qf[ni][ku]), acc[ni])
                    tok0 = blk * BLK + mt * 16 + q16 * 4
                    for ni in range_constexpr(NI):
                        v = acc[ni].load()
                        vis = qpos[ni] - tok0
                        x = [(vis >= i).select(v[i], neg_inf) for i in range_constexpr(4)]
                        m = _maxf_nn(_maxf_nn(x[0], x[1]), _maxf_nn(x[2], x[3]))
                        run[ni] = _maxf_nn(run[ni], m)

                def finish(run):
                    out = []
                    for p in range_constexpr(NI // 2):
                        x, y = _xlane_max4_pair(run[2 * p], run[2 * p + 1])
                        out += [x, y]
                    return out

                def store_group(g0, sc):
                    if const_expr(NO_STORE):
                        return
                    full = g0 + GROUP <= blk1
                    for h in range_constexpr(2):
                        row = rows[h]
                        vals = [sc[j][h] for j in range_constexpr(GROUP)]
                        for k in range_constexpr(1, NW):
                            is_k = q16 == k
                            row = is_k.select(rows[2 * k + h], row)
                            vals = [
                                is_k.select(sc[j][2 * k + h], vals[j])
                                for j in range_constexpr(GROUP)
                            ]
                        valid = row < q_len
                        off = (seq_start + row) * i32_S + g0
                        if full:
                            buffer_ops.buffer_store(
                                fx.Vector.from_elements(vals, fx.Float32), sr, off, mask=valid
                            )
                        else:
                            for j in range_constexpr(GROUP):
                                buffer_ops.buffer_store(
                                    vals[j], sr, off + j, mask=valid & (g0 + j < blk1)
                                )

                n_groups = (blk1 - blk0 + (GROUP - 1)) // GROUP
                pages = [page_of(blk0 + j) for j in range_constexpr(GROUP)]
                for gi, st in range(
                    fx.Index(0),
                    fx.Index(n_groups),
                    fx.Index(1),
                    init=[p.ir_value() for p in pages],
                ):
                    pg = [fx.Int32(st[j]) for j in range_constexpr(GROUP)]
                    g0 = blk0 + fx.Int32(gi) * GROUP
                    nxt = [page_of(g0 + GROUP + j) for j in range_constexpr(GROUP)]
                    ring = [load_pair(pg[0], p) for p in range_constexpr(RING)]
                    runs = [[neg_inf for _ in range(NI)] for _ in range(GROUP)]
                    sc = []
                    for k in range_constexpr(GROUP * PAIRS):
                        if k + RING < GROUP * PAIRS:
                            kk = k + RING
                            ring.append(load_pair(pg[kk // PAIRS], kk % PAIRS))
                        regs = ring.pop(0)
                        bj, p = k // PAIRS, k % PAIRS
                        for t in range_constexpr(2):
                            mtile(regs[t], g0 + bj, 2 * p + t, runs[bj])
                        if p == PAIRS - 1:
                            sc.append(finish(runs[bj]))
                    store_group(g0, sc)
                    res = yield [p.ir_value() for p in nxt]

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        arg_prefix: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i64_kv_bytes: fx.Int64,
        i32_nb: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q, arg_kv, arg_score, arg_bt, arg_cu, arg_seq, arg_prefix, i32_bt_stride,
            i32_S, i32_nseg, i32_qt, i32_total_q, i64_kv_bytes, i32_nb,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
