# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill index scorer, wave-private variant: no workgroup barrier.

Same contract, query tiling (4 waves x 128 rows = 512-row tile x ctx segment) and
per-wave math as ``score_prefill``. Every wave stages the K block by itself, the way
the decode scorer does: 32-token pieces (8 KB) loaded 1 KB-contiguous per instruction
(lane = row 4i + lane//16, chunk lane%16), written to the wave's own padded LDS slot
(272 B rows) and read back as MFMA A fragments. The four waves of a workgroup read
the same block within a few microseconds of each other (L1/L2 hits), and nothing
synchronises them: the LDS-staged kernel loses 20-30% of its block time to the
per-block barrier (its bar_only split). Two pieces stay in flight across the loop
(loop-carried registers), the pages of the next block group are loaded one group
ahead, scores are kept four blocks per row and stored 16 B per row.

Measured (MI355X, ctx 500K..800K, us per call at 1x2048 / 1x8192 / 4x8192 / 1x32768):
ring 2 = 352 / 1426 / 5432 / 5433, ring 1 = 427.9 / 1739 / 6575 / 6331, against 289 / 1164 /
4756 / 4892 for ``score_prefill``. Slower because every wave fetches the block itself
(4x the load requests per workgroup) and the private staging ring takes the wave to
418 VGPRs + 162 AGPRs (accumulators pushed into AGPRs; ``score_prefill`` is 308 + 52).
Lab variant only, ``score_prefill`` stays the default.
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
GROUP = 4  # blocks per loop iteration / score store
NI = ROWS_PER_WAVE // 16
KU = D // 32
HALF_TOK = 32  # tokens per staged piece (8 KB: 32 VGPRs per piece in flight)
HALVES = BLK // HALF_TOK
HALF_MT = HALF_TOK // 16
HALF_LOADS = HALF_TOK * D * 2 // 1024  # 1 KB per wave instruction
RING = int(os.environ.get("M3_IDX_PRING", "2"))  # 32-token pieces in flight (lab knob)
LDS_PAD = 16
RS = D * 2 + LDS_PAD
RS_T = RS // 16
SLOT_T = HALF_TOK * RS_T
WAVE_T = RING * SLOT_T
LDS_BYTES = NW * WAVE_T * 16  # 139264
_KNOBS = set(filter(None, os.environ.get("M3_IDX_KNOBS", "").split(",")))
K_CACHE_MOD = 2 if "knt" in _KNOBS else 0
NO_STORE = "no_store" in _KNOBS


def compile_score_prefill_wp():
    name = f"m3_index_score_prefill_bf16_wp_tq{TILE_Q}_ring{RING}" + "".join(
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
        k_cache_mod, k_no_store = K_CACHE_MOD, NO_STORE  # closure scalars: cache key
        smem = fx.SharedAllocator().allocate(Shared).peek()
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
                sr = buffer_ops.create_buffer_resource_from_addr(
                    arg_score,
                    num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4,
                )
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r for r in rows]
                # Q^T B fragments: lane (l16, q16) at K-step ku holds dims q16*32 + ku*8 .. +8
                qf = [
                    [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                qr,
                                ((seq_start + rows[ni]) * (D * 2) + q16 * 64 + ku * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        ).bitcast(fx.BFloat16)
                        for ku in range_constexpr(KU)
                    ]
                    for ni in range_constexpr(NI)
                ]
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

                bt_row = b * i32_bt_stride
                last_blk = nblk - 1

                def page_of(blk):
                    bi = (blk < nblk).select(blk, last_blk)
                    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(bt[bt_row + bi])))

                # row-contiguous loads: instruction i of a half covers rows 4i .. 4i+3
                row_off = ((lane // 16) * (D * 2) + (lane % 16) * 16) // 4
                st_tiles = [
                    (4 * i + lane // 16) * RS_T + lane % 16 for i in range_constexpr(HALF_LOADS)
                ]

                def load_half(page, h):
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
                                soffset_bytes=h * (HALF_TOK * D * 2) + i * 1024,
                            )
                        )
                        for i in range_constexpr(HALF_LOADS)
                    ]

                def stage_half(regs, slot):
                    for i in range_constexpr(HALF_LOADS):
                        lds_store16(wave_t + slot * SLOT_T + st_tiles[i], regs[i])

                mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                rd_base = l16 * RS_T + q16 * 4

                def frag8(v8):
                    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                    t.store(v8)
                    return t

                def compute_half(slot, blk, h, run):
                    a_next = [
                        lds_load16(wave_t + slot * SLOT_T + rd_base + ku).bitcast(fx.BFloat16)
                        for ku in range_constexpr(KU)
                    ]
                    for mt in range_constexpr(HALF_MT):
                        a = a_next
                        if const_expr(mt + 1 < HALF_MT):
                            a_next = [
                                lds_load16(
                                    wave_t + slot * SLOT_T + (mt + 1) * 16 * RS_T + rd_base + ku
                                ).bitcast(fx.BFloat16)
                                for ku in range_constexpr(KU)
                            ]
                        acc = [
                            fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                            for _ in range(NI)
                        ]
                        for ni in range_constexpr(NI):
                            acc[ni].store(zero4)
                        for ku in range_constexpr(KU):
                            af = frag8(a[ku])
                            for ni in range_constexpr(NI):
                                fx.gemm(mma, acc[ni], af, frag8(qf[ni][ku]), acc[ni])
                        tok0 = blk * BLK + (h * HALF_MT + mt) * 16 + q16 * 4
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
                    if const_expr(k_no_store):
                        return
                    full = g0 + GROUP <= blk1
                    for hh in range_constexpr(2):
                        row = rows[hh]
                        vals = [sc[j][hh] for j in range_constexpr(GROUP)]
                        for k in range_constexpr(1, NW):
                            is_k = q16 == k
                            row = is_k.select(rows[2 * k + hh], row)
                            vals = [
                                is_k.select(sc[j][2 * k + hh], vals[j])
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

                # groups of 4 blocks; carried: the group's 4 pages (loaded one group
                # ahead) and the RING halves in flight (the first halves of the group)
                n_groups = (blk1 - blk0 + (GROUP - 1)) // GROUP
                pages = [page_of(blk0 + j) for j in range_constexpr(GROUP)]
                ring_init = []
                for k in range_constexpr(RING):
                    ring_init += load_half(pages[k // HALVES], k % HALVES)
                init = pages + ring_init
                for gi, st in range(
                    fx.Index(0),
                    fx.Index(n_groups),
                    fx.Index(1),
                    init=[x.ir_value() for x in init],
                ):
                    pg = [fx.Int32(st[j]) for j in range_constexpr(GROUP)]
                    ring = [
                        [
                            fx.Vector(st[GROUP + k * HALF_LOADS + i])
                            for i in range_constexpr(HALF_LOADS)
                        ]
                        for k in range_constexpr(RING)
                    ]
                    g0 = blk0 + fx.Int32(gi) * GROUP
                    nxt = [page_of(g0 + GROUP + j) for j in range_constexpr(GROUP)]
                    sc = []
                    for k in range_constexpr(GROUP * HALVES):
                        kk = k + RING
                        if const_expr(kk < GROUP * HALVES):
                            ring.append(load_half(pg[kk // HALVES], kk % HALVES))
                        else:
                            kk2 = kk - GROUP * HALVES
                            ring.append(load_half(nxt[kk2 // HALVES], kk2 % HALVES))
                        regs = ring.pop(0)
                        slot = k % RING
                        stage_half(regs, slot)
                        bj, h = k // HALVES, k % HALVES
                        if const_expr(h == 0):
                            run = [neg_inf for _ in range(NI)]  # one block's running max live
                        compute_half(slot, g0 + bj, h, run)
                        if const_expr(h == HALVES - 1):
                            sc.append(finish(run))
                    store_group(g0, sc)
                    carried = list(nxt)
                    for k in range_constexpr(RING):
                        carried += ring[k]
                    res = yield [x.ir_value() for x in carried]

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
