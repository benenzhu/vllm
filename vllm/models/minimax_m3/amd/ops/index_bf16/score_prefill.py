# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill index scorer for the bf16 index cache (gfx950, one local index head).

``score[row, blk]`` = max over the 128 tokens of block ``blk`` of ``q_row . k_token``,
fp32, token positions past the query position masked to -inf; same tensor layout as
the Triton scorer (``[1, total_q, S]``, S = max_block rounded up to 16), so
``minimax_m3_index_topk`` consumes it unchanged.

A workgroup is 4 waves x 128 query rows (a 512-row query tile) x one segment of the
tile's causal block range; the grid is query tile (fastest, so concurrent workgroups
share K blocks in L2), then segment, then request. Every shape is a runtime value:
one kernel. Each wave keeps its 128 query rows in VGPRs as the MFMA B operand (Q^T:
8 N-tiles x 4 K-steps of 16 B) and walks the segment's K blocks. A block (one page:
128 tokens x 256 B = 32 KB) is loaded once per workgroup, 16 B per thread per
instruction and fully contiguous, into an LDS slot with a 272 B row stride
(conflict-free ds_read_b128), and every wave reads its 8 token M-tiles from there as
the A operand. C^T = K . Q^T puts tokens on the accumulator row axis, so a row's
block score is an in-lane max over 4 tokens x 8 M-tiles plus one 4-lane permlane
reduction per block. Scores are kept four blocks per row and stored 16 B per row
(segments are multiples of 4 blocks, so the columns are 16 B aligned); the tail
group of a segment stores single dwords.

Pipeline: two LDS slots. The next block's global loads are issued before this
block's MFMAs and written to the other slot after them: one barrier per block.
The kernel is MFMA-bound (256 MFMA 16x16x32 per block per wave); the K traffic
(one block read per query tile) is mostly L2 hits because the query tiles of a
segment run concurrently.
"""

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .utils import _global_i32_ptr, _maxf_nn, _xlane_max4_pair

NW = 4  # waves per workgroup
ROWS_PER_WAVE = 128  # 8 MFMA N-tiles of query rows
TILE_Q = NW * ROWS_PER_WAVE  # 512 query rows per workgroup
BLK = 128  # tokens per sparse block (one page)
D = 128  # index head dim
BLOCK_BYTES = BLK * D * 2
LDS_PAD = 16  # bytes of padding per LDS row: conflict-free 16 B reads
RS = D * 2 + LDS_PAD  # 272 B LDS row stride
RS_T = RS // 16  # LDS is addressed in 16 B tiles
SLOT_T = BLK * RS_T  # tiles per block slot
# lab knobs (M3_IDX_KNOBS=a,b,...): fake_k = no K loads after the first block
# (compute ceiling), no_store = no score stores, pf2 = three LDS slots with the
# loads two blocks ahead, xcd = the query tiles of a segment on one XCD, afpf =
# the next M-tile's A fragments read before this one's MFMAs
_KNOBS = set(filter(None, os.environ.get("M3_IDX_KNOBS", "").split(",")))
FAKE_K = "fake_k" in _KNOBS
NO_STORE = "no_store" in _KNOBS
PF2 = "pf2" in _KNOBS
XCD = "xcd" in _KNOBS
AFPF = "noafpf" not in _KNOBS
ACC2 = "acc2" in _KNOBS
# K-path cost split: bar_only = fake_k + a barrier per block; load_only = K loads issued and
# waited, no LDS write; write_only = LDS writes of the (stale) registers, no loads
BAR_ONLY = "bar_only" in _KNOBS
LOAD_ONLY = "load_only" in _KNOBS
WRITE_ONLY = "write_only" in _KNOBS
N_XCD = 8
N_SLOTS = 3 if PF2 else 2
LDS_BYTES = N_SLOTS * SLOT_T * 16
GROUP = 4  # blocks buffered per score store (16 B per row)
NI = ROWS_PER_WAVE // 16  # N-tiles (query rows) per wave
MT = BLK // 16  # M-tiles (tokens) per block
KU = D // 32  # K-steps per MFMA 16x16x32 over the 128 dims
CHUNKS = BLOCK_BYTES // (64 * NW * 16)  # 16 B chunks per thread per block


def compile_score_prefill():
    """One kernel for every shape (all sizes are runtime arguments)."""
    name = f"m3_index_score_prefill_bf16_tq{TILE_Q}_nw{NW}_s{N_SLOTS}" + "".join(
        "_" + k for k in sorted(_KNOBS)
    )

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # K block slots

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # idx_q [total_q, 1, 128] bf16
        arg_kv: fx.Int64,  # index cache [num_blocks, 128, 128] bf16
        arg_score: fx.Int64,  # score [1, total_q, S] fp32
        arg_bt: fx.Int64,  # block table [batch, >= max_block] i32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1] i32
        arg_seq: fx.Int64,  # seq_lens [batch] i32
        arg_prefix: fx.Int64,  # prefix_lens [batch] i32
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,  # query tiles per request
        i32_total_q: fx.Int32,
        i64_kv_bytes: fx.Int64,
        i32_nb: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        n_wg = i32_qt * i32_nseg * i32_nb
        if const_expr(XCD):
            # workgroups go round-robin over the 8 XCDs: give consecutive work ids
            # (the query tiles of one segment, which share K blocks) to one XCD
            per_xcd = (n_wg + (N_XCD - 1)) // N_XCD
            wid = (pid % N_XCD) * per_xcd + pid // N_XCD
        else:
            wid = pid
        q_tile = wid % i32_qt
        seg = (wid // i32_qt) % i32_nseg
        b = wid // (i32_qt * i32_nseg)

        bc = (wid < n_wg).select(b, fx.Int32(0))  # padded workgroups read request 0
        cu = _global_i32_ptr(arg_cu)
        seq_start = fx.Int32(cu[bc])
        q_len = fx.Int32(cu[bc + 1]) - seq_start
        seq_len = fx.Int32(_global_i32_ptr(arg_seq)[bc])
        prefix = fx.Int32(_global_i32_ptr(arg_prefix)[bc])
        bt = _global_i32_ptr(arg_bt)
        row0 = q_tile * TILE_Q
        if (wid < n_wg) & (row0 < q_len):
            # causal window of this query tile, split into segments of 4k blocks
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

                # this wave's query rows (request-local): row0 + wave*128 + ni*16 + l16;
                # rows past q_len read the next request's rows and are masked at store
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r for r in rows]  # absolute query positions
                # Q^T B fragments: lane (l16, q16) at K-step ku holds dims q16*32 + ku*8
                # .. +8 (a permutation of the dot product shared with the A operand)
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

                # LDS as 16 B tiles
                lds16 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.a.ptr),
                        fx.make_layout(LDS_BYTES // 4, 1),
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

                # block staging: thread tx moves chunk c = j*256 + tx (row c>>4, 16 B
                # column c&15); the workgroup's instruction j covers 4 KB contiguous
                st_tiles = [
                    (j * (64 * NW // 16) + tx // 16) * RS_T + tx % 16
                    for j in range_constexpr(CHUNKS)
                ]
                bt_row = b * i32_bt_stride
                last_blk = nblk - 1

                def load_block(blk):
                    bi = (blk < nblk).select(blk, last_blk)  # past the end: reread
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
                    if const_expr(LOAD_ONLY):
                        # wait for the loads (a fake dependency: keep one dword) then barrier
                        keep = regs[0][0]
                        for j in range_constexpr(1, CHUNKS):
                            keep = keep | regs[j][0]
                        if keep == 0x7EADBEEF:
                            lds_store16(slot * SLOT_T + st_tiles[0], regs[0])
                    else:
                        for j in range_constexpr(CHUNKS):
                            lds_store16(slot * SLOT_T + st_tiles[j], regs[j])
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    fx.gpu.barrier()

                mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                # A fragment of M-tile mt, K-step ku: row mt*16 + l16, bytes q16*64 + ku*16
                rd_base = l16 * RS_T + q16 * 4

                def frag8(v8):
                    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
                    t.store(v8)
                    return t

                def compute_block(slot, blk):
                    """Block scores of this wave's 128 rows (8 values per lane,
                    identical across the 4 q16 lanes)."""
                    # tokens visible to row ni: q16*4 + mt*16 + i <= qpos - blk*128
                    vis = [qpos[ni] - (blk * BLK + q16 * 4) for ni in range_constexpr(NI)]
                    run = [neg_inf for _ in range(NI)]

                    def read_a(mt):
                        return [
                            lds_load16(slot * SLOT_T + mt * 16 * RS_T + rd_base + ku).bitcast(
                                fx.BFloat16
                            )
                            for ku in range_constexpr(KU)
                        ]

                    accs = [
                        [
                            fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                            for _ in range(NI)
                        ]
                        for _ in range(2 if ACC2 else 1)
                    ]

                    def issue(mt, a):
                        acc = accs[mt % len(accs)]
                        for ni in range_constexpr(NI):
                            acc[ni].store(zero4)
                        for ku in range_constexpr(KU):
                            af = frag8(a[ku])
                            for ni in range_constexpr(NI):
                                fx.gemm(mma, acc[ni], af, frag8(qf[ni][ku]), acc[ni])

                    def reduce(mt):
                        acc = accs[mt % len(accs)]
                        for ni in range_constexpr(NI):
                            v = acc[ni].load()
                            for i in range_constexpr(4):
                                x = (vis[ni] >= mt * 16 + i).select(v[i], neg_inf)
                                run[ni] = _maxf_nn(run[ni], x)

                    if const_expr(AFPF):
                        a_next = read_a(0)
                    for mt in range_constexpr(MT):
                        if const_expr(AFPF):
                            a = a_next
                            if const_expr(mt + 1 < MT):
                                a_next = read_a(mt + 1)
                        else:
                            a = read_a(mt)
                        issue(mt, a)
                        if const_expr(ACC2):
                            # the previous M-tile's max/mask runs under these MFMAs
                            if const_expr(mt >= 1):
                                reduce(mt - 1)
                        else:
                            reduce(mt)
                    if const_expr(ACC2):
                        reduce(MT - 1)
                    out = []
                    for p in range_constexpr(NI // 2):
                        x, y = _xlane_max4_pair(run[2 * p], run[2 * p + 1])
                        out += [x, y]
                    return out

                def store_group(g0, sc):
                    """sc[j][ni]: scores of blocks g0..g0+3. Lane q16 = k stores rows
                    ni = 2k, 2k+1, 16 B per row (single dwords for a partial tail)."""
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
                                fx.Vector.from_elements(vals, fx.Float32),
                                sr,
                                off,
                                mask=valid,
                            )
                        else:
                            for j in range_constexpr(GROUP):
                                buffer_ops.buffer_store(
                                    vals[j], sr, off + j, mask=valid & (g0 + j < blk1)
                                )

                n_groups = (blk1 - blk0 + (GROUP - 1)) // GROUP
                if const_expr(FAKE_K or BAR_ONLY):
                    # compute ceiling: every block computed from slot 0 (no K traffic)
                    stage_block(load_block(blk0), 0)
                    for g in range(0, n_groups):
                        g0 = blk0 + fx.Int32(g) * GROUP
                        sc = []
                        for j in range_constexpr(GROUP):
                            sc.append(compute_block(0, g0 + j))
                            if const_expr(BAR_ONLY):
                                fx.gpu.barrier()
                        store_group(g0, sc)
                elif const_expr(WRITE_ONLY):
                    regs0 = load_block(blk0)
                    stage_block(regs0, 0)
                    for g in range(0, n_groups):
                        g0 = blk0 + fx.Int32(g) * GROUP
                        sc = []
                        for j in range_constexpr(GROUP):
                            sc.append(compute_block(j % N_SLOTS, g0 + j))
                            stage_block(regs0, (j + 1) % N_SLOTS)
                        store_group(g0, sc)
                elif const_expr(PF2):
                    # three slots, slot(blk) = (blk - blk0) % 3; blocks g0, g0+1 staged at
                    # group start, loads issued two blocks ahead (the last one of a group
                    # one ahead), one barrier per block
                    stage_block(load_block(blk0), 0)
                    stage_block(load_block(blk0 + 1), 1)
                    for g in range(0, n_groups):
                        g0 = blk0 + fx.Int32(g) * GROUP
                        s0 = (fx.Int32(g) * GROUP) % N_SLOTS  # slot of g0
                        sc = []
                        pend = []
                        for j in range_constexpr(GROUP):
                            blk = g0 + j
                            pend.append((load_block(blk + 2), (s0 + j + 2) % N_SLOTS))
                            sc.append(compute_block((s0 + j) % N_SLOTS, blk))
                            if const_expr(j >= 1):
                                regs, slot = pend.pop(0)
                                stage_block(regs, slot)
                        regs, slot = pend.pop(0)
                        stage_block(regs, slot)
                        store_group(g0, sc)
                else:
                    stage_block(load_block(blk0), 0)
                    for g in range(0, n_groups):
                        g0 = blk0 + fx.Int32(g) * GROUP
                        sc = []
                        for j in range_constexpr(GROUP):
                            blk = g0 + j
                            nxt = load_block(blk + 1)  # in flight during this block's MFMAs
                            sc.append(compute_block(j % N_SLOTS, blk))
                            stage_block(nxt, (j + 1) % N_SLOTS)
                        store_group(g0, sc)

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
            arg_q,
            arg_kv,
            arg_score,
            arg_bt,
            arg_cu,
            arg_seq,
            arg_prefix,
            i32_bt_stride,
            i32_S,
            i32_nseg,
            i32_qt,
            i32_total_q,
            i64_kv_bytes,
            i32_nb,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
