# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny bf16 GEMM ``out[M, N] = x[M, K] @ w[N, K]^T`` for decode batches (M <= 128).

At decode the dense projections (o_proj: K 2048 -> N 6144 per TP4 rank, the fused
qkv/index projections: K 6144) are weight-streaming: 25-28 MB of bf16 weights
per call against a few hundred KB of activations, and hipBLASLt runs them at
2.4-2.6 TB/s. This kernel streams the weight once at (close to) HBM speed:

* W is preshuffled once (``host.shuffle_weight``) into 1 KB blocks of 16 columns x
  32 K, laid out as the B fragment of ``v_mfma_f32_16x16x32_bf16``: lane
  ``klane*16 + n`` holds K ``klane*8 .. +8`` of column ``n``. One buffer load of
  16 B per lane is one MFMA K-step, and a wave's loads walk 1 KB blocks
  contiguously through memory (non-temporal, the weight is read once).
* A (L2/MALL resident: it is tiny and every workgroup reads it) is staged through
  LDS in batches of 4 K-steps (256 B per row): one load instruction covers 4 rows
  x 256 B, so at most 4 rows 4 KB apart share an L2 channel; loading the MFMA
  fragment directly (16 rows x 64 B per instruction) puts all 16 rows on one
  channel and costs ~2 us per row tile. Each K-wave owns its LDS slot (no
  barrier when the 4 waves are K-waves). Rows beyond M read as zero through the
  buffer resource's record count.
* workgroup = ``NW`` waves owning a ``TILE_N``-column strip of the output;
  ``KW`` K-waves split K (the partial sums meet in LDS, K-wave 0 adds and
  stores), ``NW/KW`` N-waves split the columns. Every wave keeps all ``RT =
  ceil(M/16)`` row tiles of its columns in accumulators, so the weight fragment
  it loads feeds ``RT`` MFMAs.
* ``KS`` workgroups split K on top of that (grid ``N/TILE_N x KS``): every
  workgroup must ingest all of A for its K range, and the per-CU ingest rate
  (~32 B/clk) is the wall from M ~ 32 up (knockouts 09-13: the global A loads
  are 3.7 of 10.8 us at M = 80), so K is spread over more CUs. The K-split
  workgroups write fp32 partials to a scratch buffer with device-scope (sc1,
  write-through) stores, bump a per-strip counter (agent-scope atomic), and the
  last one to arrive reads the partials with device-scope loads, sums and stores
  bf16: no spinning, so no co-residency requirement. Release/acquire fences
  would do the same through ``buffer_wbl2``/``buffer_inv`` and cost 13 us per
  launch (measured 09-13).
* ``PF`` W tiles in flight per wave.
* Where the time goes (o_proj shape, M = 80, 10.3 us, knockouts of 09-13): the
  launch and prologue ~2.9 us, the weight stream ~3 us, the A ingest ~3 us (per
  CU, ~30 B/clk: it halves with the K-split, but the hand-off through memory
  costs ~2.7 us, so o_proj does not split), MFMA + LDS ~3.7 us at M = 80; they
  overlap only partly.

The tile constants are plain scalars captured by the kernel closure: FlyDSL's
disk cache keys a kernel by its source plus the scalar closure values, so
constants hidden in an object would make every variant share one binary.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

NW = 4  # waves per workgroup
TILE_K = 32  # K per W block (1 KB = 16 columns x 32 K bf16 = one MFMA K-step)
MAX_RT = 8  # row tiles: M <= 128
# (TILE_N, K-waves, W prefetch, K-split workgroups); per shape in host.tile_for
# (MI355X lab sweeps of 09-13).
DEFAULT_TILE = (32, 4, 4, 1)


def compile_skinny(*, N, K, RT, TILE_N=DEFAULT_TILE[0], KW=DEFAULT_TILE[1], prefetch=DEFAULT_TILE[2], KS=DEFAULT_TILE[3], dbg=0):
    """Kernel for ``ceil(M/16) == RT`` row tiles of a ``[*, K] x [N, K]^T`` GEMM.
    ``launch.kernel_name`` identifies the compiled kernel, ``launch.grid`` is the
    workgroup count, ``launch.scratch`` = (fp32 partial elements, counter
    elements) the host must provide when ``KS > 1`` (``host.scratch_for``).
    ``dbg`` bits are lab knockouts (wrong results): 1 = unmasked stores, 2 =
    unbounded output resource, 4 = no global A loads, 8 = no LDS fragment reads,
    16 = plain (non-coherent) partials, 32 = no arrival counter, 64 = no W loads.
    Knockout numbers of 09-13 (o_proj shape, M = 80, 10.3 us): no W 7.8, no A 7.1,
    neither 6.6 (MFMA + LDS + launch), nothing but the launch ~2.9."""
    assert NW % KW == 0 and 1 <= RT <= MAX_RT
    NWN = NW // KW  # N-waves
    NPW = TILE_N // NWN  # columns per wave
    NI = NPW // 16  # 16-column tiles per wave
    assert NPW % 16 == 0 and N % TILE_N == 0
    KT = K // TILE_K  # W blocks per column group
    assert K % (KS * KW * TILE_K) == 0
    KT_WG = KT // KS  # W blocks per workgroup
    KTW = KT_WG // KW  # W blocks per K-wave
    PF = min(prefetch, KTW)
    KB = 4  # K-steps per A batch (256 B per row)
    assert KTW % KB == 0
    NBATCH = KTW // KB
    RS = KB * 64 + 16  # LDS row stride: 256 B + 16 B pad (conflict-free 16 B reads)
    RS_T = RS // 16
    SLOT = RT * 16 * RS  # one K-wave's A batch
    SLOT_T = SLOT // 16
    # K-reduce scratch (reuses the A slots): K-waves 1.. park RT x NI accumulator tiles
    RED_BYTES = (KW - 1) * RT * NWN * NI * 1024
    RED_SLOT = RT * NWN * NI * 64  # 16 B tiles parked per K-wave
    LDS_BYTES = max(KW * SLOT, RED_BYTES, 16)
    assert LDS_BYTES <= 160 * 1024, LDS_BYTES
    NB = N // TILE_N  # column strips
    PSLOT = NWN * RT * NI * 256  # fp32 partial of one workgroup, in the accumulator layout
    PART_ELEMS = NB * KS * PSLOT
    CNT_STRIDE = 32  # dwords: one arrival counter per 128 B line (same-line atomics serialize)
    W_BYTES = N * K * 2
    assert W_BYTES <= 0xFFFFFFFF, "buffer resources address 4 GB"
    w_cache_mod = 2  # non-temporal: the weight is read once per call

    @fx.struct
    class Shared:
        a: fx.Array[fx.Uint8, LDS_BYTES, 16]  # A slots / K-reduce scratch
        flag: fx.Array[fx.Int32, 4]  # the K-split arrival count, broadcast

    name = (
        f"m3_skinny_a16w16_n{N}_k{K}_rt{RT}_tn{TILE_N}_kw{KW}_pf{PF}"
        + (f"_ks{KS}" if KS > 1 else "")
        + (f"_dbg{dbg}" if dbg else "")
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_out: fx.Int64, i32_m: fx.Int32, arg_part: fx.Int64, arg_cnt: fx.Int64):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16, q16 = lane % 16, lane // 16
        wave_n, wave_k = wave % NWN, wave // NWN
        if const_expr(KS > 1):
            nb, ks = pid // KS, pid % KS  # the K-splits of a strip are dispatched together
        else:
            nb, ks = pid, 0
        # K-wave k owns the k-th KTW blocks of the workgroup's K range (rotating the
        # walk per workgroup or interleaving the K-waves to spread A over more L2
        # channels measured nothing, 09-13)
        kblk0 = ks * KT_WG + wave_k * KTW  # this wave's first W block / A K-step
        a_soff = [b * 256 for b in range_constexpr(NBATCH)]
        w_soff = [b * 4096 for b in range_constexpr(NBATCH)]

        # LDS as 16 B tiles (A slots, then the K-reduce scratch)
        lds16 = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Int32, smem.a.ptr), fx.make_layout(LDS_BYTES // 4, 1)
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

        # rows >= M are out of the record range: their A fragments read as zero
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x, num_records_bytes=fx.Int64(i32_m) * (K * 2)
        )
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=W_BYTES)
        if const_expr(dbg & 2):
            outr = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=0xFFFFFFFF)
        else:
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=fx.Int64(i32_m) * (N * 2)
            )

        # A batch load: instruction (rt, j) reads rows rt*16 + j*4 + q16, 16 B piece
        # l16 of the batch's 256 B (K wave_k*KTW*32 + b*128 + l16*8); b*256 B goes
        # into soffset. Staged at LDS row rt*16 + j*4 + q16, tile l16.
        a_dw = [
            [
                ((rt * 16 + j * 4 + q16) * K + kblk0 * TILE_K + l16 * 8) // 2
                for j in range_constexpr(4)
            ]
            for rt in range_constexpr(RT)
        ]
        a_slot = wave_k * SLOT_T
        a_st = [[(rt * 16 + j * 4 + q16) * RS_T + l16 for j in range_constexpr(4)] for rt in range_constexpr(RT)]
        # fragment of row tile rt, K-step ku of the batch: row rt*16 + l16, tile ku*4 + q16
        a_rd = [(rt * 16 + l16) * RS_T + q16 for rt in range_constexpr(RT)]

        def load_a_batch(b):
            if const_expr(dbg & 4):  # knockout: no global A traffic (lane-dependent junk)
                return [
                    [fx.Vector.filled(4, lane + b * 7 + rt * 3 + j, fx.Int32) for j in range_constexpr(4)]
                    for rt in range_constexpr(RT)
                ]
            return [
                [
                    fx.Vector(
                        buffer_ops.buffer_load(
                            xr, a_dw[rt][j], vec_width=4, dtype=fx.Int32, soffset_bytes=a_soff[b]
                        )
                    )
                    for j in range_constexpr(4)
                ]
                for rt in range_constexpr(RT)
            ]

        def stage_a_batch(regs):
            for rt in range_constexpr(RT):
                for j in range_constexpr(4):
                    lds_store16(a_slot + a_st[rt][j], regs[rt][j])
            if const_expr(NWN > 1):  # a wave's own LDS ops are in order; other waves' need the barrier
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()

        def read_a(rt, ku):
            if const_expr(dbg & 8):  # knockout: no LDS fragment reads
                return fx.Vector.filled(4, lane + ku * 5 + rt, fx.Int32).bitcast(fx.BFloat16)
            return lds_load16(a_slot + a_rd[rt] + ku * 4).bitcast(fx.BFloat16)
        # W block (column group ng, K-step kt): byte (ng*KT + kt)*1024 + lane*16;
        # this wave's K range starts at block kblk0; within a batch (kt%4)*256 dwords
        # in the immediate, the batch's 4 KB in soffset
        nbase = nb * TILE_N + wave_n * NPW
        w_dw = [
            (((nbase + ni * 16) // 16) * KT + kblk0) * 256 + lane * 4
            for ni in range_constexpr(NI)
        ]

        def load_tile(kt):
            if const_expr(dbg & 64):  # knockout: no W traffic
                return [fx.Vector.filled(4, lane + kt * 3 + ni, fx.Int32) for ni in range_constexpr(NI)]
            return [
                fx.Vector(
                    buffer_ops.buffer_load(
                        wr,
                        w_dw[ni] + (kt % KB) * 256,
                        vec_width=4,
                        dtype=fx.Int32,
                        cache_modifier=w_cache_mod,
                        soffset_bytes=w_soff[kt // KB],
                    )
                )
                for ni in range_constexpr(NI)
            ]

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        acc = [
            [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(NI)]
            for _ in range(RT)
        ]
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        for rt in range_constexpr(RT):
            for ni in range_constexpr(NI):
                acc[rt][ni].store(zero4)

        def _frag(v8):
            t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            t.store(v8)
            return t

        def read_frags(ku):
            return [read_a(rt, ku) for rt in range_constexpr(RT)]

        def compute_tile(bw, frags):
            b_t = [_frag(bw[ni].bitcast(fx.BFloat16)) for ni in range_constexpr(NI)]
            for rt in range_constexpr(RT):
                a_t = _frag(frags[rt])
                for ni in range_constexpr(NI):
                    fx.gemm(mma_atom, acc[rt][ni], a_t, b_t[ni], acc[rt][ni])

        # pipeline: the next A batch in flight in registers, the current one in LDS;
        # W ring PF tiles deep; the A fragments one K-step ahead in registers. Per
        # K-step: (batch b + 1's loads at batch b's first step) W load, the next
        # step's LDS fragment reads, this step's MFMAs; batch b + 1 is staged after
        # batch b's last step (this wave's own slot). The sched_barrier per K-step
        # keeps the loads where they are issued: left alone, the scheduler sinks
        # every load to just before its MFMA to save registers and each K-step
        # pays a full memory latency (16.7 us instead of ~10 at M = 80). Two A
        # batches in flight measured nothing.
        abuf = load_a_batch(0)
        ring = []
        for t in range_constexpr(PF):
            ring.append(load_tile(t))
        stage_a_batch(abuf)
        frags = read_frags(0)
        for kt in range_constexpr(KTW):
            b, ku = kt // KB, kt % KB
            if const_expr(ku == 0 and b + 1 < NBATCH):
                abuf = load_a_batch(b + 1)
            if const_expr(kt + PF < KTW):
                ring.append(load_tile(kt + PF))
            fx.rocdl.sched_barrier(0)
            bw = ring.pop(0)
            nfrags = read_frags(ku + 1) if const_expr(ku + 1 < KB) else None
            compute_tile(bw, frags)
            if const_expr(ku == KB - 1 and b + 1 < NBATCH):
                if const_expr(NWN > 1):
                    fx.gpu.barrier()  # the slot's readers are done
                stage_a_batch(abuf)
                nfrags = read_frags(0)
            frags = nfrags

        if const_expr(KW > 1):
            # K-reduce (scratch = the A slots, so every wave must be done reading them):
            # K-wave k > 0 parks its partial sums in slot k - 1, K-wave 0 adds them and
            # stores alone
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            red = [
                [((rt * NWN + wave_n) * NI + ni) * 64 + lane for ni in range_constexpr(NI)]
                for rt in range_constexpr(RT)
            ]
            if wave_k > 0:
                park = (wave_k - 1) * RED_SLOT
                for rt in range_constexpr(RT):
                    for ni in range_constexpr(NI):
                        lds_store16(red[rt][ni] + park, acc[rt][ni].load().bitcast(fx.Int32))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            if wave_k == 0:
                for rt in range_constexpr(RT):
                    for ni in range_constexpr(NI):
                        v = acc[rt][ni].load()
                        for kw in range_constexpr(KW - 1):
                            pv = lds_load16(red[rt][ni] + kw * RED_SLOT).bitcast(fx.Float32)
                            v = fx.Vector.from_elements(
                                [v[i] + pv[i] for i in range_constexpr(4)], fx.Float32
                            )
                        acc[rt][ni].store(v)

        # epilogue: lane (q16, l16) holds rows rt*16 + q16*4 + ii of column l16
        def epilogue():
            for rt in range_constexpr(RT):
                for ii in range_constexpr(4):
                    row = rt * 16 + q16 * 4 + ii
                    valid = row < i32_m
                    for ni in range_constexpr(NI):
                        yb = acc[rt][ni].load()[ii].to(fx.BFloat16)
                        if const_expr(dbg & 1):
                            buffer_ops.buffer_store(yb, outr, row * N + nbase + ni * 16 + l16)
                        else:
                            buffer_ops.buffer_store(yb, outr, row * N + nbase + ni * 16 + l16, mask=valid)

        if const_expr(KS > 1):
            # K-split reduction across workgroups. Partial slot (nb, ks): the
            # accumulator layout, 16 B per lane per (wave_n, rt, ni).
            partr = buffer_ops.create_buffer_resource_from_addr(
                arg_part, num_records_bytes=PART_ELEMS * 4
            )
            pdw = [
                [((wave_n * RT + rt) * NI + ni) * 256 + lane * 4 for ni in range_constexpr(NI)]
                for rt in range_constexpr(RT)
            ]
            my_slot = (nb * KS + ks) * PSLOT
            coh = 0 if dbg & 16 else 16  # sc1: device scope (knockout 16 = plain, wrong)
            if wave_k == 0:
                for rt in range_constexpr(RT):
                    for ni in range_constexpr(NI):
                        buffer_ops.buffer_store(
                            acc[rt][ni].load(), partr, my_slot + pdw[rt][ni], cache_modifier=coh
                        )
                # the partial has reached memory before this workgroup's arrival is counted
                fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            flag = smem.flag.ptr
            if const_expr(dbg & 32):  # knockout (wrong results): no arrival counter
                arrived = ks
            else:
                if tx == 0:
                    cnt_ptr = fx.inttoptr(
                        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
                        fx.Int64(arg_cnt),
                    )
                    old = fx.Int32(
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.add,
                            (cnt_ptr + nb * CNT_STRIDE).llvm_ptr,
                            fx.Int32(1).ir_value(),
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=4,
                        ).result
                    )
                    flag[0] = old
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                arrived = fx.Int32(flag[0])
            # the counter only ever grows (launches add exactly KS per strip), the
            # last of the KS arrivals sums every partial (its own included) and stores
            if arrived % KS == KS - 1:
                if wave_k == 0:
                    for rt in range_constexpr(RT):
                        for ni in range_constexpr(NI):
                            v = zero4
                            for o in range_constexpr(KS):
                                pv = fx.Vector(
                                    buffer_ops.buffer_load(
                                        partr,
                                        (nb * KS + o) * PSLOT + pdw[rt][ni],
                                        vec_width=4,
                                        dtype=fx.Int32,
                                        cache_modifier=coh,
                                    )
                                ).bitcast(fx.Float32)
                                v = fx.Vector.from_elements(
                                    [v[i] + pv[i] for i in range_constexpr(4)], fx.Float32
                                )
                            acc[rt][ni].store(v)
                    epilogue()
        elif const_expr(KW > 1):
            if wave_k == 0:
                epilogue()
        else:
            epilogue()

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_out: fx.Int64, i32_m: fx.Int32, arg_part: fx.Int64, arg_cnt: fx.Int64, stream: fx.Stream):
        kernel(arg_x, arg_w, arg_out, i32_m, arg_part, arg_cnt).launch(
            grid=(NB * KS, 1, 1), block=(64 * NW, 1, 1), stream=stream
        )

    launch.kernel_name = name
    launch.grid = NB * KS
    launch.tile = (TILE_N, KW, PF, KS)
    launch.scratch = (PART_ELEMS, NB * CNT_STRIDE) if KS > 1 else (0, 0)
    return launch
