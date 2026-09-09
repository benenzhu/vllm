# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill index scorer, 8-wave variant: a 1024-row query tile per workgroup.

Same contract and per-wave structure as ``score_prefill`` (128 query rows per wave
in VGPRs as the MFMA B operand, K blocks through LDS as the A operand, C^T = K . Q^T,
in-lane + 4-lane max per block). Differences:

- 8 waves share every K block, so a block is read from memory once per 1024 query
  rows instead of once per 512: half the K traffic, which is what bounds the
  multi-request prefill steps (their K working set does not fit the infinity cache).
  Two waves per SIMD also cover each other's MFMA drains.
- The block loop is a runtime loop with one block per iteration (the next block's
  loads issued before this block's MFMAs, staged after them, one barrier per block).
- Block scores go to a per-wave LDS tile (128 rows x 16 blocks, 80 B row stride:
  conflict-free scalar writes and 16 B reads) and are flushed every 16 blocks as
  64 B per row (single masked dwords for a segment's tail), so the score stores are
  full 64 B segments and no score registers stay live across blocks.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .utils import _global_i32_ptr, _maxf_nn, _xlane_max4_pair

NW = int(os.environ.get("M3_IDX_NW", "8"))  # waves per workgroup (lab knob)
ROWS_PER_WAVE = int(os.environ.get("M3_IDX_RPW", "128"))  # MFMA N-tiles x 16 of query rows (lab knob)
TILE_Q = NW * ROWS_PER_WAVE  # 1024 query rows per workgroup
BLK = 128
D = 128
BLOCK_BYTES = BLK * D * 2
LDS_PAD = 16
RS = D * 2 + LDS_PAD  # 272 B K row stride
RS_T = RS // 16
SLOT_T = BLK * RS_T  # 16 B tiles per K slot
N_SLOTS = 2
K_BYTES = N_SLOTS * SLOT_T * 16  # 69632
GROUP = 16  # blocks per score flush (64 B per row)
SROW_DW = 20  # score tile row stride in dwords (80 B): conflict-free, 16 B aligned
S_DW_PER_WAVE = ROWS_PER_WAVE * SROW_DW  # dwords per wave score tile
S_BYTES = NW * S_DW_PER_WAVE * 4  # 81920
NI = ROWS_PER_WAVE // 16
MT = BLK // 16
KU = D // 32
CHUNKS = BLOCK_BYTES // (64 * NW * 16)  # 16 B chunks per thread per block
PER_LANE = NI // 4  # score rows kept per q16 lane
assert NI % 4 == 0
_KNOBS = set(filter(None, os.environ.get("M3_IDX_KNOBS", "").split(",")))
FAKE_K = "fake_k" in _KNOBS
NO_STORE = "no_store" in _KNOBS
AFPF = "noafpf" not in _KNOBS
assert K_BYTES + S_BYTES <= 160 * 1024, (K_BYTES, S_BYTES)


def compile_score_prefill_w8():
    name = f"m3_index_score_prefill_bf16_tq{TILE_Q}_nw{NW}_rpw{ROWS_PER_WAVE}_s{N_SLOTS}_g{GROUP}" + "".join(
        "_" + k for k in sorted(_KNOBS)
    )

    @fx.struct
    class Shared:
        k: fx.Array[fx.Uint8, K_BYTES, 16]  # K block slots
        s: fx.Array[fx.Uint8, S_BYTES, 16]  # per-wave score tiles

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
                kr = buffer_ops.create_buffer_resource_from_addr(
                    arg_kv, num_records_bytes=i64_kv_bytes
                )
                sr = buffer_ops.create_buffer_resource_from_addr(
                    arg_score,
                    num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4,
                )
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r for r in rows]
                qf = [
                    [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                qr,
                                ((seq_start + rows[ni]) * (D * 2) + q16 * 64 + ku * 16)
                                // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        ).bitcast(fx.BFloat16)
                        for ku in range_constexpr(KU)
                    ]
                    for ni in range_constexpr(NI)
                ]

                lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

                def tiles16(ptr, nbytes):
                    return fx.logical_divide(
                        fx.make_view(
                            fx.recast_iter(fx.Int32, ptr), fx.make_layout(nbytes // 4, 1)
                        ),
                        fx.make_layout(4, 1),
                    )

                k16 = tiles16(smem.k.ptr, K_BYTES)
                s16 = tiles16(smem.s.ptr, S_BYTES)
                s32 = fx.recast_iter(fx.Int32, smem.s.ptr)  # dword pointer

                def lds_store16(view, tile, vec4):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(vec4)
                    fx.copy(lds_atom, r, fx.slice(view, (None, tile)))

                def lds_load16(view, tile):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(lds_atom, fx.slice(view, (None, tile)), r)
                    return r.load()

                # K staging: thread tx moves chunk c = j*512 + tx (row c>>4, column c&15)
                st_tiles = [
                    (j * (64 * NW // 16) + tx // 16) * RS_T + tx % 16
                    for j in range_constexpr(CHUNKS)
                ]
                bt_row = b * i32_bt_stride
                last_blk = nblk - 1

                def load_block(blk):
                    bi = (blk < nblk).select(blk, last_blk)
                    page = fx.Int32(bt[bt_row + bi])
                    base = page * (BLOCK_BYTES // 4)
                    return [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                kr,
                                base + (j * (64 * NW) + tx) * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        for j in range_constexpr(CHUNKS)
                    ]

                def stage_block(regs, slot):
                    for j in range_constexpr(CHUNKS):
                        lds_store16(k16, slot * SLOT_T + st_tiles[j], regs[j])
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    fx.gpu.barrier()

                mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                rd_base = l16 * RS_T + q16 * 4

                def frag8(v8):
                    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                    t.store(v8)
                    return t

                def compute_block(slot, blk):
                    vis = [qpos[ni] - (blk * BLK + q16 * 4) for ni in range_constexpr(NI)]
                    run = [neg_inf for _ in range(NI)]

                    def read_a(mt):
                        return [
                            lds_load16(k16, slot * SLOT_T + mt * 16 * RS_T + rd_base + ku).bitcast(
                                fx.BFloat16
                            )
                            for ku in range_constexpr(KU)
                        ]

                    if const_expr(AFPF):
                        a_next = read_a(0)
                    for mt in range_constexpr(MT):
                        if const_expr(AFPF):
                            a = a_next
                            if const_expr(mt + 1 < MT):
                                a_next = read_a(mt + 1)
                        else:
                            a = read_a(mt)
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
                        for ni in range_constexpr(NI):
                            v = acc[ni].load()
                            for i in range_constexpr(4):
                                x = (vis[ni] >= mt * 16 + i).select(v[i], neg_inf)
                                run[ni] = _maxf_nn(run[ni], x)
                    out = []
                    for p in range_constexpr(NI // 2):
                        x, y = _xlane_max4_pair(run[2 * p], run[2 * p + 1])
                        out += [x, y]
                    return out

                # score tile: lane q16 = k keeps rows ni = k*PER_LANE + h of its wave
                s_wave = wave * S_DW_PER_WAVE
                s_rows = []
                s_tile_t = wave * (S_DW_PER_WAVE // 4)
                for h in range_constexpr(PER_LANE):
                    r = h * 16 + l16  # ni = h for k = 0
                    for k in range_constexpr(1, 4):
                        r = (q16 == k).select((k * PER_LANE + h) * 16 + l16, r)
                    s_rows.append(r)

                def score_to_lds(sc, col):
                    for h in range_constexpr(PER_LANE):
                        v = sc[h]
                        for k in range_constexpr(1, 4):
                            v = (q16 == k).select(sc[k * PER_LANE + h], v)
                        s32[s_wave + s_rows[h] * SROW_DW + col] = v.bitcast(fx.Int32)

                def flush(g0, ncols):
                    """rows of this wave, columns g0 .. g0+ncols (<= 16): lane reads 16 B
                    (4 columns) of row lane//4 + 16*step and stores it (64 B per row)."""
                    if const_expr(NO_STORE):
                        return
                    chunk = lane % 4
                    full = ncols >= GROUP
                    for step in range_constexpr(ROWS_PER_WAVE // 16):
                        r = step * 16 + lane // 4
                        vec4 = lds_load16(s16, s_tile_t + r * (SROW_DW // 4) + chunk)
                        grow = row0 + wave * ROWS_PER_WAVE + r
                        valid = grow < q_len
                        off = (seq_start + grow) * i32_S + g0 + chunk * 4
                        if full:
                            buffer_ops.buffer_store(
                                vec4.bitcast(fx.Float32), sr, off, mask=valid
                            )
                        else:
                            for j in range_constexpr(4):
                                buffer_ops.buffer_store(
                                    vec4.bitcast(fx.Float32)[j],
                                    sr,
                                    off + j,
                                    mask=valid & (chunk * 4 + j < ncols),
                                )

                if const_expr(FAKE_K):
                    stage_block(load_block(blk0), 0)
                    for bi in range(blk0, blk1):
                        blk = fx.Int32(bi)
                        col = (blk - blk0) % GROUP
                        score_to_lds(compute_block(0, blk), col)
                        if (col == GROUP - 1) | (blk == blk1 - 1):
                            flush(blk - col, col + 1)
                else:
                    stage_block(load_block(blk0), 0)
                    for bi in range(blk0, blk1):
                        blk = fx.Int32(bi)
                        slot = (blk - blk0) % N_SLOTS
                        nxt = load_block(blk + 1)  # in flight during this block's MFMAs
                        sc = compute_block(slot, blk)
                        col = (blk - blk0) % GROUP
                        score_to_lds(sc, col)
                        stage_block(nxt, 1 - slot)
                        if (col == GROUP - 1) | (blk == blk1 - 1):
                            flush(blk - col, col + 1)

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
