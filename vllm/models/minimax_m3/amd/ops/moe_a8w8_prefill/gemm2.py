# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 prefill MoE stage 2 for MXFP8 (a8w8, down projection), token-major
bf16 output:

    out[tok*topk + slot, :] = bf16( (h[row, :] @ W2[e]^T) * sorted_weights[row] )

reduced by aiter's ``moe_reduction_kernel`` like production stage 2.

Structure: the fp8 gemm1 mainloop (4-wave 2x2, 8-buffer LDS ping-pong, depth-2
K pipeline of 128-K steps, ``v_mfma_scale_f32_16x16x128_f8f6f4``, AGPR
accumulators) run as one flat sequence over the CTA's n-tiles: a CTA owns an
m-tile of 128 sorted rows and sweeps ``NT`` n-tiles of 256 W2 rows (``n_split``
CTAs share the 6144 columns; rotated start inside long same-expert runs as in
the a4w4 kernel). K = I = 768 is 6 steps per n-tile; the loads for the first two
steps of n-tile t+1 are issued during steps 4/5 of n-tile t, so the DMA stream
never drains. A (16 KB per step) is re-read from L2 per n-tile: the fp8 A tile
(96 KB) does not fit next to two B stages in LDS, and pinned in AGPRs it would
take 192 of them.

Epilogue: after the last MFMA of an n-tile the 128 accumulators are scaled by
the row's routing weight, packed to bf16 and stored token-major (8 B per lane
per tile, non-temporal). Every store is in bounds: the padded rows (tok ==
n_tokens) go to ``topk`` scratch rows appended to OUT.

Scale blocks (256 B = 32 rows x 8 K-groups): 3 per operand per n-tile
(K = 768), 6 LDS slots; block g of n-tile t sits in slot (3t + g) % 6, gathered
three steps ahead (step 1: block 2 of t; steps 3/5: blocks 0/1 of t+1).

Layouts (bytes):
  A         [num_m_blocks*BM, I]         sorted rows, fp8 (gemm1 OUT_Q)
  A_scale   [pad32(num_m_blocks*BM), I/32]  sorted rows, e8m0-shuffled (gemm1 OUT_sc)
  W2        [E, H, I]                    aiter shuffle_weight (16 x 64 K blocks)
  W2_sc     [E*H/32, I/256, 4, 16] dwords aiter shuffle_scale
  sorted_w  [num_m_blocks*BM]            f32 routing weight per sorted row
  OUT       [n_tokens*topk + topk, H]    bf16 (the last topk rows: scratch)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill.gemm1 import (
    _N_WAVES,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_SLOT_BYTES,
    G2SLoaderAsm,
    S2RLoaderFp4,
    ScaleGatherMoE,
    ScaleLoaderLDS,
    _Buf,
    _divmod_nonneg,
    _flat_frag,
    _g2s_thunks,
    _riffle,
    _s2r_thunks,
    _swizzled_col,
    _unflat_frag,
    wait_barrier,
)
from vllm.models.minimax_m3.amd.ops.moe_a4w4_prefill.gemm2 import (
    _NUM_XCDS,
    _STORE_CPOL,
    _bf16x2,
    _v2i32,
)

from .gemm1 import BLOCK_K, Mfma16x16x128Fp8

_SCALE_SLOTS2 = 6


def compile_moe_gemm2(
    *,
    H: int,
    I: int,  # noqa: E741
    E: int,
    topk: int,
    n_split: int = 2,
    sort_block_m: int = 128,
):
    """Grouped fp8 gemm2 for one (H, I, E, topk). ``sort_block_m`` (128 / 256) is
    the ``moe_sorting`` block of the inputs; the kernel tiles 128 rows and skips the
    all-padding halves of a 256-sort."""
    BM = 128
    BN = 256
    K = I
    K_BYTES = K
    BLOCK_K_BYTES = BLOCK_K
    EID_SHIFT = (sort_block_m // BM).bit_length() - 1  # 0 or 1
    assert sort_block_m in (BM, 2 * BM)
    N_TILES_ALL = H // BN
    assert H % BN == 0 and N_TILES_ALL % n_split == 0
    NT = N_TILES_ALL // n_split  # n-tiles per CTA
    K_ITERS = K // BLOCK_K
    assert K % 256 == 0 and K_ITERS == 6, "the flat schedule is written for K = 768"
    LDS_BLOCK_M = BM // 2
    LDS_BLOCK_N = BN // 2
    N_TILES_A = LDS_BLOCK_M // 2 // 16  # 2
    N_TILES_B = LDS_BLOCK_N // 2 // 16  # 4
    N_ACCUMS = N_TILES_A * N_TILES_B
    NB = N_TILES_B
    NA = N_TILES_A
    S_STORES = 4 * N_ACCUMS  # 8-B stores per lane per n-tile (4 quadrants x 8 tiles)
    OUT_ROW_BYTES = H * 2
    W2_BYTES = E * H * K
    assert W2_BYTES <= 0xFFFFFFFF
    B_TILE_BYTES = BN * K_BYTES  # one n-tile of W2 (256 rows)

    a_lds_size = LDS_BLOCK_M * BLOCK_K_BYTES  # 8 KB
    b_lds_size = LDS_BLOCK_N * BLOCK_K_BYTES  # 16 KB
    A_BUFS = 4 * a_lds_size
    LDS_TILES_BYTES = A_BUFS + 4 * b_lds_size  # 96 KB
    SCALE_LDS_BYTES2 = _SCALE_SLOTS2 * _SCALE_SLOT_BYTES  # 24 KB

    A_WAVE_GROUPS = N_TILES_A // 2  # 1
    A_HALF_GROUPS = LDS_BLOCK_M // 32  # 2
    B_WAVE_GROUPS = 2  # 32-row groups per wave (64 W2 rows)
    B_HALF_GROUPS = LDS_BLOCK_N // 32  # 4: half 1 = 128 rows further
    SC_COLS = K // 32
    W2_SC_BYTES = E * H * SC_COLS

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, LDS_TILES_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SCALE_LDS_BYTES2, 16]

    @flyc.kernel
    def kernel_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr
        _scale_base_ptr = lds.scale_lds.ptr

        a_cur0 = _Buf(_base_ptr, 0 * a_lds_size)
        a_cur1 = _Buf(_base_ptr, 1 * a_lds_size)
        a_next0 = _Buf(_base_ptr, 2 * a_lds_size)
        a_next1 = _Buf(_base_ptr, 3 * a_lds_size)
        b_cur0 = _Buf(_base_ptr, A_BUFS + 0 * b_lds_size)
        b_cur1 = _Buf(_base_ptr, A_BUFS + 1 * b_lds_size)
        b_next0 = _Buf(_base_ptr, A_BUFS + 2 * b_lds_size)
        b_next1 = _Buf(_base_ptr, A_BUFS + 3 * b_lds_size)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        g4 = lane_id // 16
        r16 = lane_id % 16

        # ---- work item: (m-tile, n-chunk); consecutive m-tiles share an XCD ----
        nv_rsrc = buffer_ops.create_buffer_resource(
            num_valid_ids, max_size=False, num_records_bytes=4
        )
        num_valid = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        n_work = (num_valid // fx.Int32(BM)) * fx.Int32(n_split)
        per_xcd = (n_work + fx.Int32(_NUM_XCDS - 1)) // fx.Int32(_NUM_XCDS)
        intra, xcd = _divmod_nonneg(fx.block_idx.x, _NUM_XCDS)
        work = xcd * per_xcd + intra
        block_valid = (intra < per_xcd) & (work < n_work)
        work_safe = block_valid.select(work, fx.Int32(0))
        tile_i, chunk = _divmod_nonneg(work_safe, n_split)
        m_base = tile_i * BM
        eid_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
        )
        expert = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                tile_i >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_ids, max_size=False, num_records_bytes=num_m_blocks * (BM * 4)
        )
        if const_expr(EID_SHIFT > 0):
            # a 256-sort pads each expert to 256 rows: a 128-row tile whose first
            # row is the sentinel (tok == n_tokens) holds nothing to compute
            first_sid = fx.Int32(
                buffer_ops.buffer_load(
                    ids_rsrc, m_base, vec_width=1, dtype=fx.Int32, is_scalar=True
                )
            )
            block_valid = block_valid & ((first_sid & fx.Int32(0x00FFFFFF)) < n_tokens)
        chunk_n0 = chunk * NT  # first n-tile (global index) of this CTA
        # ---- rotated n-tile sweep inside runs of >= 4 same-expert m-tiles (see the
        # a4w4 kernel: neighbouring CTAs then find their next W2 tile in L2) ----
        ROT_STRIDE, ROT_GATE = 2, 3
        _d = fx.Int32(ROT_GATE)
        _lo_ok = tile_i >= _d
        _hi_ok = tile_i + _d < num_m_blocks
        _e_lo = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                _lo_ok.select(tile_i - _d, fx.Int32(0)) >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        _e_hi = fx.Int32(
            buffer_ops.buffer_load(
                eid_rsrc,
                _hi_ok.select(tile_i + _d, fx.Int32(0)) >> EID_SHIFT,
                vec_width=1,
                dtype=fx.Int32,
                is_scalar=True,
            )
        )
        rot_on = (_lo_ok & (_e_lo == expert)) | (_hi_ok & (_e_hi == expert))
        nt_rot = rot_on.select(
            _divmod_nonneg(tile_i * fx.Int32(ROT_STRIDE), NT)[1], fx.Int32(0)
        )

        def _pn(nt):
            """CTA-local logical n-tile (sweep order) -> physical n-tile of the chunk"""
            x = nt + nt_rot
            return (x >= fx.Int32(NT)).select(x - fx.Int32(NT), x)

        def _n_glob(nt):
            """global n-tile index of logical n-tile ``nt`` (clamped to the sweep:
            the redundant loads after the last tile re-fetch the last one)"""
            ntc = (nt >= fx.Int32(NT)).select(fx.Int32(NT - 1), nt)
            return chunk_n0 + _pn(ntc)

        if block_valid:
            sw_rsrc = buffer_ops.create_buffer_resource(
                sorted_weights,
                max_size=False,
                num_records_bytes=num_m_blocks * (BM * 4),
            )
            a_rsrc = buffer_ops.create_buffer_resource(
                A, max_size=False, num_records_bytes=num_m_blocks * (BM * K_BYTES)
            )
            b_rsrc = buffer_ops.create_buffer_resource(
                W2, max_size=False, num_records_bytes=W2_BYTES
            )
            # OUT has topk scratch rows after the n_tokens*topk real ones: the padded
            # sorted rows (tok == n_tokens) and the prologue's dummy stores land there,
            # so every store is in bounds and retires in issue order (a dropped
            # out-of-bounds store completes early and breaks the vmcnt schedule)
            out_rsrc = buffer_ops.create_buffer_resource(
                OUT,
                max_size=False,
                num_records_bytes=(n_tokens + fx.Int32(1)) * (topk * OUT_ROW_BYTES),
            )

            # ---- A rows (sorted, contiguous): row * K_BYTES + swizzled col ----
            def _a_offsets(half):
                offs = []
                for rnd in range_constexpr(N_TILES_A):
                    row = lane_id // 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id % 8) * 16
                    grow = m_base + half * LDS_BLOCK_M + row
                    offs.append(grow * fx.Int32(K_BYTES) + _swizzled_col(row, col))
                return offs

            gl_off_a0 = _a_offsets(0)
            gl_off_a1 = _a_offsets(1)

            # ---- W2: LDS row r of half hf = W2 row expert*H + n*256 + hf*128 + r,
            # 16-row blocks of K_BYTES*16 bytes, a8 preshuffle inside; the n-tile
            # goes in soffset (B_TILE_BYTES per tile) ----
            e_row0 = expert * fx.Int32(H)

            def _b_offsets(hf):
                offs = []
                for rnd in range_constexpr(N_TILES_B):
                    row = lane_id % 8 + wave_id * 8 + rnd * (_N_WAVES * 8)
                    col = (lane_id // 8) * 16
                    wrow = e_row0 + fx.Int32(hf * LDS_BLOCK_N + row)
                    offs.append(
                        (wrow // fx.Int32(16)) * fx.Int32(K_BYTES * 16)
                        + fx.Int32(
                            (row % 16) * 16
                            + (col // 64) * 1024
                            + ((col % 64) // 16) * 256
                            + (col % 16)
                        )
                    )
                return offs

            gl_off_b0 = _b_offsets(0)
            gl_off_b1 = _b_offsets(1)
            A_K_STEP = BLOCK_K_BYTES
            B_K_STEP = 2 * 1024

            mfma = Mfma16x16x128Fp8(N_TILES_A, N_TILES_B)
            sb_index = lambda j, hf: (j // 2, j % 2)  # noqa: E731  (32-row groups)

            scale_gather = ScaleGatherMoE(
                A_scale,
                W2_scale,
                K,
                lane_id,
                wave_id,
                _scale_base_ptr,
                num_m_blocks * (BM * SC_COLS),
                W2_SC_BYTES,
                A_WAVE_GROUPS,
                A_HALF_GROUPS,
                B_WAVE_GROUPS,
                B_HALF_GROUPS,
            )

            def _set_scale_tile(nt):
                scale_gather.set_wave_base(
                    m_base, e_row0 + _n_glob(nt) * fx.Int32(BN)
                )

            a_scale_ld = ScaleLoaderLDS(
                N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
            )
            b_scale_ld = ScaleLoaderLDS(
                N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
            )

            def _slot(nt, g):
                return (fx.Int32(nt) * fx.Int32(3) + fx.Int32(g)) % fx.Int32(_SCALE_SLOTS2)

            a0_g2s = G2SLoaderAsm(a_rsrc, gl_off_a0, N_TILES_A, wave_id)
            a1_g2s = G2SLoaderAsm(a_rsrc, gl_off_a1, N_TILES_A, wave_id)
            b0_g2s = G2SLoaderAsm(b_rsrc, gl_off_b0, N_TILES_B, wave_id)
            b1_g2s = G2SLoaderAsm(b_rsrc, gl_off_b1, N_TILES_B, wave_id)
            for ld in (a0_g2s, a1_g2s, b0_g2s, b1_g2s):
                ld.set_wave_base(_base_ptr)
            a_s2r = S2RLoaderFp4(wave_i, N_TILES_A)
            b_s2r = S2RLoaderFp4(wave_j, N_TILES_B)

            # ---- output rows / weights of the 4 rows this lane writes ----
            def _orow(sid):
                tok = sid & fx.Int32(0x00FFFFFF)  # padded: tok == n_tokens -> scratch
                slot = (sid >> 24) & fx.Int32(0xFF)
                return tok * fx.Int32(topk) + slot

            out_off = []
            row_w = []
            for h in range_constexpr(2):
                for ti in range_constexpr(N_TILES_A):
                    row = m_base + fx.Int32(h * LDS_BLOCK_M) + wave_i * (N_TILES_A * 16) + ti * 16 + r16
                    sid = fx.Int32(
                        buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=fx.Int32)
                    )
                    out_off.append(_orow(sid) * fx.Int32(OUT_ROW_BYTES))
                    row_w.append(
                        fx.Float32(
                            buffer_ops.buffer_load(
                                sw_rsrc, row, vec_width=1, dtype=fx.Float32
                            )
                        )
                    )
            # ---- prologue = "steps 4 and 5 of n-tile -1": the same issue order as
            # the steady state (step 4: a0 b0 b1 a1; step 5: a0 b0 b1 gather a1),
            # block 0's gather up front ----
            _set_scale_tile(fx.Int32(0))
            scale_gather.gather(0, _slot(0, 0))
            b_soff0 = _n_glob(fx.Int32(0)) * fx.Int32(B_TILE_BYTES)
            a0_g2s.load(a_cur0, fx.Int32(0 * A_K_STEP))
            b0_g2s.load(b_cur0, b_soff0 + fx.Int32(0 * B_K_STEP))
            b1_g2s.load(b_cur1, b_soff0 + fx.Int32(0 * B_K_STEP))
            a1_g2s.load(a_cur1, fx.Int32(0 * A_K_STEP))
            a0_g2s.load(a_next0, fx.Int32(1 * A_K_STEP))
            b0_g2s.load(b_next0, b_soff0 + fx.Int32(1 * B_K_STEP))
            b1_g2s.load(b_next1, b_soff0 + fx.Int32(1 * B_K_STEP))
            scale_gather.gather(1, _slot(0, 1))
            a1_g2s.load(a_next1, fx.Int32(1 * A_K_STEP))

            # gather 0 + a_cur0 landed: everything younger may fly
            wait_barrier((3 * NA) + (4 * NB) + 1)
            a0_frag = a_s2r.load(a_cur0)
            # b_cur0 and b_cur1 landed
            wait_barrier((3 * NA) + (2 * NB) + 1)
            b0_frag = b_s2r.load(b_cur0, preshuffled=True)
            b1_frag = b_s2r.load(b_cur1, preshuffled=True)
            sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0, 0))
            sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0, 0))
            sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)

            # Per step kc (0..5) of n-tile nt, in issue order: a0 (NA), b0 (NB),
            # [SEG2] b1 (NB), scale gather (odd kc), a1 (NA), all for flat step
            # 6*nt + kc + 2; after step 5's MFMAs the S_STORES epilogue stores.
            # Loop-top wait: the step before the previous one complete; SEG2: the
            # previous step's a0/b0/b1 landed. The counts are the loads issued after
            # the last one needed; the epilogue stores are not counted: stores and
            # loads retire out of order with respect to each other (LLVM's
            # SIInsertWaitcnts: mixed pending events), so a count that included the
            # stores as younger ops could pass with the needed loads still in flight
            # (measured: races in the first n-tile). The price is that step 0's top
            # wait of the next tile drains the stores.
            def _top_vmcnt(kc):
                return 2 * NA + 2 * NB + (1 - kc % 2)

            def _seg2_vmcnt(kc):
                return 2 * NA + NB + (1 - kc % 2)

            def _read_scale_thunks(nt, kc_next, holder):
                # scales of flat step +1: same tile block kc_next//2, or block 0 of
                # the next tile (Python ints only: an ``if`` here would become an
                # scf.if and lose the binding)
                s = _slot(nt + fx.Int32(kc_next // K_ITERS), (kc_next % K_ITERS) // 2)

                def _r(dst, ld, half, _s=s):
                    holder[dst] = ld.read_half(_s, half)

                return [
                    lambda: _r(0, a_scale_ld, 0),
                    lambda: _r(1, a_scale_ld, 1),
                    lambda: _r(2, b_scale_ld, 0),
                    lambda: _r(3, b_scale_ld, 1),
                ]

            def _one_step(nt, kc, a0f, b0f, b1f_in, sc, accs, bufs):
                """``nt`` loop value (logical n-tile), ``kc`` Python int 0..5."""
                k2 = kc % 2
                ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
                saR0, saR1, sbC0, sbC1 = sc
                c00f, c01f, c10f, c11f = accs
                zero_acc = kc == 0

                _a1 = [None] * NA
                _a0n = [None] * NA
                _b0n = [None] * NB
                _b1n = [None] * NB
                # loads for flat step kc + 2
                kn = (kc + 2) % K_ITERS
                nt_ld = nt if kc + 2 < K_ITERS else nt + fx.Int32(1)
                a_off = fx.Int32(kn * A_K_STEP)
                b_off = _n_glob(nt_ld) * fx.Int32(B_TILE_BYTES) + fx.Int32(kn * B_K_STEP)

                _scn = [None, None, None, None]
                _rd_scn = _read_scale_thunks(nt, kc + 1, _scn)
                # gathers (odd steps): step 1 -> (nt, 2); 3 -> (nt+1, 0); 5 -> (nt+1, 1)
                _g_blk = {1: 2, 3: 0, 5: 1}.get(kc, 1)
                _g_slot = _slot(nt + fx.Int32({1: 0}.get(kc, 1)), _g_blk)
                _sc_gather = [lambda: scale_gather.gather(_g_blk, _g_slot)] * k2

                wait_barrier(_top_vmcnt(kc))
                il = (
                    _riffle(
                        _g2s_thunks(a0_g2s, ac0, a_off, NA),
                        _s2r_thunks(a_s2r, ac1, _a1, NA, False),
                    )
                    + _rd_scn[:2]
                )
                c00f = mfma.call(
                    a0f, b0f, c00f, saR0, sbC0, 0, k2, interleave=il, zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                il = _riffle(_g2s_thunks(b0_g2s, bc0, b_off, NB), _rd_scn[2:])
                c01f = mfma.call(
                    a0f, b1f_in, c01f, saR0, sbC1, 1, k2, interleave=il, zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                a1f = _a1

                wait_barrier(_seg2_vmcnt(kc))
                il = (
                    _riffle(
                        _g2s_thunks(b1_g2s, bc1, b_off, NB),
                        _s2r_thunks(a_s2r, an0, _a0n, NA, False),
                    )
                    + _sc_gather
                )
                c10f = mfma.call(
                    a1f, b0f, c10f, saR1, sbC0, 0, k2, interleave=il, zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                a0nf = _a0n
                il = _riffle(
                    _g2s_thunks(a1_g2s, ac1, a_off, NA),
                    _s2r_thunks(b_s2r, bn0, _b0n, NB, True)
                    + _s2r_thunks(b_s2r, bn1, _b1n, NB, True),
                )
                c11f = mfma.call(
                    a1f, b1f_in, c11f, saR1, sbC1, 1, k2, interleave=il, zero_acc=zero_acc,
                    sb_index=sb_index,
                )
                sc_next = (_scn[0], _scn[1], _scn[2], _scn[3])
                new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
                return a0nf, _b0n, _b1n, sc_next, (c00f, c01f, c10f, c11f), new_bufs

            def _epilogue(nt, accs):
                """routing-weighted bf16 rows of n-tile ``nt``, token-major"""
                col0 = (_n_glob(nt) * fx.Int32(BN) + wave_j * (N_TILES_B * 16)) * 2
                for q in range_constexpr(4):  # c00, c01, c10, c11
                    h, hf = q // 2, q % 2
                    for ti in range_constexpr(NA):
                        w = row_w[h * NA + ti]
                        for j in range_constexpr(NB):
                            v = Vec(accs[q][mfma.idx(ti, j)])
                            d0 = _bf16x2(fx.Float32(v[0]) * w, fx.Float32(v[1]) * w)
                            d1 = _bf16x2(fx.Float32(v[2]) * w, fx.Float32(v[3]) * w)
                            off = out_off[h * NA + ti] + col0 + fx.Int32(
                                (hf * LDS_BLOCK_N + j * 16) * 2
                            ) + g4 * fx.Int32(8)
                            buffer_ops.buffer_store(
                                _v2i32(d0, d1),
                                out_rsrc,
                                off,
                                offset_is_bytes=True,
                                cache_modifier=_STORE_CPOL,
                            )

            bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)
            n_a = 2 * NA
            n_b = 2 * NB
            n_sc = 2 * (NA // 2) + 2 * (NB // 2)
            _R = fx.as_ir_value

            def _flat_sc(sc):
                return [_R(v) for part in sc for v in part]

            def _unflat_sc(flat):
                sizes = (NA // 2, NA // 2, NB // 2, NB // 2)
                out, o = [], 0
                for n in sizes:
                    out.append(list(flat[o : o + n]))
                    o += n
                return tuple(out)

            init_state = _flat_frag(a0_frag) + _flat_frag(b0_frag) + _flat_frag(b1_frag) + _flat_sc(sc0)
            for nt_idx, state in range(0, NT, 1, init=init_state):
                nt = fx.Int32(nt_idx)  # the loop value is an index; i32 everywhere below
                off = 0
                a0f = _unflat_frag(state[off : off + n_a], NA)
                off += n_a
                b0f = _unflat_frag(state[off : off + n_b], NB)
                off += n_b
                b1f = _unflat_frag(state[off : off + n_b], NB)
                off += n_b
                sc = _unflat_sc(state[off : off + n_sc])
                accs = ([None] * N_ACCUMS, [None] * N_ACCUMS, [None] * N_ACCUMS, [None] * N_ACCUMS)
                bufs = bufs0
                # the gather of step 1 (block 2 of this tile) needs this tile's scale
                # base; from step 2 on the gathers belong to n-tile nt + 1. Both are
                # (re)computed from the loop value: the loop body is emitted once.
                _set_scale_tile(nt)
                for kc in range_constexpr(K_ITERS):
                    if const_expr(kc == 2):
                        _set_scale_tile(nt + fx.Int32(1))
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(nt, kc, a0f, b0f, b1f, sc, accs, bufs)
                _epilogue(nt, accs)
                state = yield _flat_frag(a0f) + _flat_frag(b0f) + _flat_frag(b1f) + _flat_sc(sc)
            # the redundant loads of the tile after the last one must land before
            # the LDS is handed on
            wait_barrier(0)

    @flyc.jit
    def launch_gemm2(
        A: fx.Tensor,
        W2: fx.Tensor,
        OUT: fx.Tensor,
        A_scale: fx.Tensor,
        W2_scale: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        n_tokens: fx.Int32,
        num_m_blocks: fx.Int32,
        grid_size: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm2(
            A,
            W2,
            OUT,
            A_scale,
            W2_scale,
            sorted_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            n_tokens,
            num_m_blocks,
            grid_size,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_size, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


def gemm2_grid(num_m_blocks: int, n_split: int) -> int:
    """blocks to launch: every (m-tile, chunk) of the allocation, rounded to the 8
    XCDs (the kernel drops the fully padded tail tiles at runtime)."""
    return (num_m_blocks * n_split + _NUM_XCDS - 1) // _NUM_XCDS * _NUM_XCDS
