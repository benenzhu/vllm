# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill index scorer for the fp8 (e4m3) index cache: the fp8 port of the bf16
``index_bf16.score_prefill_dma`` kernel, a drop-in for AITER's
``pa_sparse_block_score_prefill`` (same score buffer, causal masking and
init/local sentinels, so AITER's top-k consumes the result unchanged).

Tiling: 4 waves x 128 query rows = a 512-row query tile x a segment of the
request's context; Q^T resident in VGPRs (8 x i32 per 16-row N-tile), one
``v_mfma_f32_16x16x128_f8f6f4`` per (16-token M-tile, N-tile) -- the whole
128-dim head in one instruction, 64 MFMAs per K block per wave against 256 for
bf16 -- one barrier per K block, scores 16 B per row per 4 blocks.

A K block (128 tokens x 128 B = 16 KB) reaches LDS by ``buffer_load_dwordx4 ...
lds`` DMA, 4 x 1 KB per wave, the block b+1 in flight while block b computes.
Rows are XOR-swizzled: the 16 B column ``c`` of row ``r`` sits at
``c ^ ((r >> 1) & 7)``. With 128 B rows the row parity already picks the upper
or lower 32 banks, so the swizzle only has to spread 8 rows of one parity over
the 8 columns; the 16 rows of an MFMA A fragment then hit 16 distinct bank
groups. Page ids come from a scalar buffer load one block ahead (SMEM, so its
wait never drains the DMA queue). Two accumulator sets and a pinned per-M-tile
schedule: the previous tile's max runs under this tile's MFMAs.

The fp8 MFMA wants per lane the K bytes ``[16*klane, +16)`` and
``[64 + 16*klane, +16)`` of its row as one 8-VGPR operand (klane = lane // 16),
the same permutation for the A (K) and B (Q^T) operands, so it cancels in the
dot product.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .utils import (
    _global_i32_ptr,
    _maxf_nn,
    _mfma_fp8_16x16x128,
    _pack8,
    _xlane_max4_pair,
)

NW = 4  # waves per workgroup
ROWS_PER_WAVE = 128  # 8 MFMA N-tiles of query rows
TILE_Q = NW * ROWS_PER_WAVE  # 512 query rows per workgroup
BLK = 128  # tokens per sparse block (one page)
D = 128  # index head dim
ROW_BYTES = D  # 128 B per fp8 token row
BLOCK_BYTES = BLK * ROW_BYTES  # 16 KB
RS_T = ROW_BYTES // 16  # 16 B tiles per LDS row
SLOT_T = BLK * RS_T  # tiles per block slot
N_SLOTS = 2  # slot of block b = b % 2 (compile-time inside a 4-block group)
PF = 1  # blocks of DMA in flight ahead of the one being computed
LDS_BYTES = N_SLOTS * BLOCK_BYTES  # 32 KB
GROUP = 4  # blocks buffered per score store (16 B per row)
NI = ROWS_PER_WAVE // 16  # N-tiles (query rows) per wave
MT = BLK // 16  # M-tiles (tokens) per block
DMA_PER_WAVE = BLOCK_BYTES // (
    NW * 64 * 16
)  # 4 x 1 KB DMA instructions per wave per block
SENTINEL_INIT = 1e30  # AITER's forced-selection scores (its top-k orders by them)
SENTINEL_LOCAL = 1e29


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _asm_void(operands, asm, constraints, clobbers=""):
    """Side-effecting void inline asm (LLVM sees no memory op, so it inserts no
    waitcnt for it: the DMA completion wait is explicit)."""
    if clobbers:
        constraints = f"{constraints},{clobbers}"
    llvm.inline_asm(
        None, [_ir(o) for o in operands], asm, constraints, has_side_effects=True
    )


def compile_score_prefill():
    """One kernel for every shape (all sizes are runtime arguments)."""
    name = f"m3_index_score_prefill_fp8_dma_tq{TILE_Q}_nw{NW}"

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # K block slots

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # idx_q [total_q, 1, 128] e4m3
        arg_kv: fx.Int64,  # index cache [num_blocks, 128, 128] e4m3
        arg_score: fx.Int64,  # score [1, total_q, S] fp32
        arg_bt: fx.Int64,  # block table [batch, >= max_block] i32
        arg_cu: fx.Int64,  # cu_seqlens_q [batch + 1] i32
        arg_seq: fx.Int64,  # seq_lens [batch] i32
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,  # query tiles per request
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i64_bt_bytes: fx.Int64,  # block table storage size (rows may be padded)
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        n_wg = i32_qt * i32_nseg * i32_nb
        wid = pid
        q_tile = wid % i32_qt
        seg = (wid // i32_qt) % i32_nseg
        b = wid // (i32_qt * i32_nseg)

        bc = (wid < n_wg).select(b, fx.Int32(0))  # padded workgroups read request 0
        cu = _global_i32_ptr(arg_cu)
        seq_start = fx.Int32(cu[bc])
        q_len = fx.Int32(cu[bc + 1]) - seq_start
        seq_len = fx.Int32(_global_i32_ptr(arg_seq)[bc])
        prefix = seq_len - q_len  # rows are aligned to the end of the request
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
                    arg_q, num_records_bytes=fx.Int64(i32_total_q) * ROW_BYTES
                )
                sr = buffer_ops.create_buffer_resource_from_addr(
                    arg_score,
                    num_records_bytes=fx.Int64(i32_total_q) * fx.Int64(i32_S) * 4,
                )

                # this wave's query rows (request-local): row0 + wave*128 + ni*16 +
                # l16; rows past q_len read the next request's rows and are masked
                # at store
                rows = [row0 + wave * ROWS_PER_WAVE + ni * 16 + l16 for ni in range(NI)]
                qpos = [prefix + r for r in rows]  # absolute query positions
                # Q^T B operands: lane (l16, q16) holds dims [16*q16, +16) and
                # [64 + 16*q16, +16) of its row (the fp8 MFMA's K permutation)
                qf = []
                for ni in range_constexpr(NI):
                    qrow = (seq_start + rows[ni]) * ROW_BYTES
                    halves = [
                        fx.Vector(
                            buffer_ops.buffer_load(
                                qr,
                                (qrow + h * 64 + q16 * 16) // 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        for h in range_constexpr(2)
                    ]
                    qf.append(_pack8(halves[0], halves[1]))

                # LDS as 16 B tiles
                lds16 = fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, smem.a.ptr),
                        fx.make_layout(LDS_BYTES // 4, 1),
                    ),
                    fx.make_layout(4, 1),
                )
                lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

                def lds_load16(tile):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(lds_atom, fx.slice(lds16, (None, tile)), r)
                    return r.load()

                # DMA: instruction jj of wave w fills LDS bytes (w*4 + jj) * 1 KB ..
                # +1 KB of the slot = rows r = (w*4 + jj)*8 + lane//8, 16 B column
                # lane%8.
                # Column c of row r holds the block's column c ^ ((r >> 1) & 7), and
                # (r >> 1) & 7 = ((jj & 1) << 2) | lane//16 for these rows.
                lds_base_s = fx.Int32(
                    fx.rocdl.readfirstlane(T.i32, fx.Int32(fx.ptrtoint(smem.a.ptr)))
                )
                soff0 = fx.Int32(fx.rocdl.readfirstlane(T.i32, wave * 0))
                dma_voff = []
                for jj in range_constexpr(DMA_PER_WAVE):
                    sw = ((jj & 1) << 2) | q16
                    dma_voff.append(
                        (wave * DMA_PER_WAVE + jj) * 1024
                        + (lane >> 3) * ROW_BYTES
                        + (((lane & 7) ^ sw) << 4)
                    )
                bt_row = b * i32_bt_stride
                last_blk = nblk - 1

                btr = buffer_ops.create_buffer_resource_from_addr(
                    arg_bt, num_records_bytes=i64_bt_bytes
                )

                def page_of(blk):
                    """page id of block ``blk`` (past the end: the last block): a scalar
                    buffer load (SMEM, lgkmcnt) the compiler schedules and waits for; a
                    vector load here would sit in vmcnt and its wait would drain the
                    DMA"""
                    bi = (blk < nblk).select(blk, last_blk)
                    return fx.Int32(
                        buffer_ops.buffer_load(
                            btr,
                            bt_row + bi,
                            vec_width=1,
                            dtype=fx.Int32,
                            is_scalar=True,
                        )
                    )

                def dma_block(page, slot):
                    """issue the wave's 4 DMA loads of block ``page`` into ``slot``"""
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_kv + fx.Int64(page) * BLOCK_BYTES,
                        num_records_bytes=BLOCK_BYTES,
                    )
                    for jj in range_constexpr(DMA_PER_WAVE):
                        m0 = lds_base_s + (
                            slot * BLOCK_BYTES + (wave * DMA_PER_WAVE + jj) * 1024
                        )
                        _asm_void(
                            [m0, dma_voff[jj], kr, soff0],
                            "s_mov_b32 m0, $0\n"
                            "buffer_load_dwordx4 $1, $2, $3 offen lds",
                            "s,v,s,s",
                        )

                def block_done():
                    """block b+1 has landed (only the newest PF-1 blocks of DMA may
                    still be in flight), all waves past their reads of block b"""
                    fx.rocdl.s_waitcnt(vmcnt=DMA_PER_WAVE * (PF - 1), lgkmcnt=0)
                    fx.gpu.barrier()

                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                neg_inf = fx.Float32(float("-inf"))
                # A operand of M-tile mt: row mt*16 + l16, 16 B columns q16 and 4 + q16,
                # swizzled by (row >> 1) & 7 = l16 >> 1
                rd_row = l16 * RS_T
                rd_col = [q16 ^ (l16 >> 1), (4 + q16) ^ (l16 >> 1)]

                def compute_block(slot, blk, masked):
                    """Block scores of this wave's 128 rows (8 values per lane,
                    identical across the 4 q16 lanes). ``masked`` (compile-time):
                    apply the causal select; False for blocks every row of the tile
                    sees whole."""
                    if const_expr(masked):
                        vis = [
                            qpos[ni] - (blk * BLK + q16 * 4)
                            for ni in range_constexpr(NI)
                        ]
                    run = [neg_inf for _ in range(NI)]

                    def read_a(mt):
                        base = slot * SLOT_T + mt * 16 * RS_T + rd_row
                        return _pack8(
                            lds_load16(base + rd_col[0]), lds_load16(base + rd_col[1])
                        )

                    # two accumulator sets: the max of M-tile mt-1 runs under the MFMAs
                    # of
                    # M-tile mt
                    accs = [[None] * NI, [None] * NI]

                    def issue(mt, a):
                        acc = accs[mt % 2]
                        for ni in range_constexpr(NI):
                            acc[ni] = _mfma_fp8_16x16x128(a, qf[ni], zero4)

                    def reduce(mt):
                        acc = accs[mt % 2]
                        for ni in range_constexpr(NI):
                            v = acc[ni]
                            for i in range_constexpr(4):
                                x = v[i]
                                if const_expr(masked):
                                    x = (vis[ni] >= mt * 16 + i).select(x, neg_inf)
                                run[ni] = _maxf_nn(run[ni], x)

                    a_next = read_a(0)
                    for mt in range_constexpr(MT):
                        a = a_next
                        if const_expr(mt + 1 < MT):
                            a_next = read_a(mt + 1)
                        issue(mt, a)
                        if const_expr(mt >= 1):
                            reduce(mt - 1)
                        # pin this tile's schedule: one MFMA then the previous tile's
                        # share of max (plus the causal selects when masked), the next
                        # tile's A operand reads under the first MFMAs. Left to itself
                        # the scheduler consumes accumulators right behind their MFMAs
                        # (it frees registers) and pays the MFMA -> VALU hazard every
                        # tile.
                        n_valu = 16 if not masked else 48
                        for i in range_constexpr(NI):
                            fx.rocdl.sched_group_barrier(0x008, 1, 0)
                            fx.rocdl.sched_group_barrier(
                                0x002, (n_valu + NI - 1) // NI, 0
                            )
                            if const_expr(i < 2):
                                fx.rocdl.sched_group_barrier(0x100, 1, 0)
                        fx.rocdl.sched_barrier(0)
                    reduce(MT - 1)
                    out = []
                    for p in range_constexpr(NI // 2):
                        x, y = _xlane_max4_pair(run[2 * p], run[2 * p + 1])
                        out += [x, y]
                    return out

                sent_init = fx.Float32(SENTINEL_INIT)
                sent_local = fx.Float32(SENTINEL_LOCAL)

                def store_group(g0, sc):
                    """sc[j][ni]: scores of blocks g0..g0+3. Lane q16 = k stores rows
                    ni = 2k, 2k+1, 16 B per row (single dwords for a partial tail).
                    The row's init / local blocks get AITER's sentinels instead."""
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
                        nb_row = (
                            prefix + row
                        ) // BLK + 1  # the row's causal block count
                        for j in range_constexpr(GROUP):
                            blk = g0 + j
                            is_local = (blk >= nb_row - i32_local) & (blk < nb_row)
                            is_init = blk < i32_init
                            vals[j] = is_local.select(
                                sent_local, is_init.select(sent_init, vals[j])
                            )
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
                # groups whose 4 blocks every row of the tile sees whole (block
                # blk*128+127 <= prefix + row0) need no causal select
                v = prefix + row0 - (BLK - 1)
                last_full = (v < 0).select(fx.Int32(-1), v // BLK)
                d = last_full - (GROUP - 1) - blk0
                nf = (d < 0).select(fx.Int32(0), d // GROUP + 1)
                n_full = (nf < n_groups).select(nf, n_groups)

                # prologue: blocks blk0 .. blk0+PF-1 in flight, the page of blk0+PF
                # (the DMA issued during block blk0) loaded; wait for all of them once
                pgs = [page_of(blk0 + i) for i in range_constexpr(PF)]
                for i in range_constexpr(PF):
                    dma_block(pgs[i], i % N_SLOTS)
                pg_next = page_of(blk0 + PF)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                fx.gpu.barrier()

                def group(g0, pg_next, masked):
                    """blocks g0 .. g0+3: per block, issue the DMA of block b+PF (its
                    page fetched one block earlier), compute, wait for block b+1,
                    barrier. Returns the page of the next block's DMA."""
                    sc = []
                    for j in range_constexpr(GROUP):
                        blk = g0 + j
                        dma_block(
                            pg_next, (j + PF) % N_SLOTS
                        )  # page loaded a block ago
                        pg_next = page_of(blk + PF + 1)
                        sc.append(compute_block(j % N_SLOTS, blk, masked))
                        block_done()
                    store_group(g0, sc)
                    return pg_next

                def one(st):
                    return fx.Int32(st[0] if isinstance(st, (list, tuple)) else st)

                for g, st in range(
                    fx.Index(0), fx.Index(n_full), fx.Index(1), init=[_ir(pg_next)]
                ):
                    pg_next = group(blk0 + fx.Int32(g) * GROUP, one(st), False)
                    res = yield [_ir(pg_next)]
                for g, st in range(
                    fx.Index(n_full),
                    fx.Index(n_groups),
                    fx.Index(1),
                    init=[_ir(one(res))],
                ):
                    pg_next = group(blk0 + fx.Int32(g) * GROUP, one(st), True)
                    res = yield [_ir(pg_next)]

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_cu: fx.Int64,
        arg_seq: fx.Int64,
        i32_bt_stride: fx.Int32,
        i32_S: fx.Int32,
        i32_nseg: fx.Int32,
        i32_qt: fx.Int32,
        i32_total_q: fx.Int32,
        i32_nb: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
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
            i32_bt_stride,
            i32_S,
            i32_nseg,
            i32_qt,
            i32_total_q,
            i32_nb,
            i64_bt_bytes,
            i32_init,
            i32_local,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
