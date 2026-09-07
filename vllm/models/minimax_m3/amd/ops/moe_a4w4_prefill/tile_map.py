# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""gemm1's expert-major tile order as a device kernel (replaces the ~75 us torch
construction in the benches; one CTA, a few microseconds).

Sorted rows come in expert order, so the valid m-blocks of an expert are one
contiguous run ``[lo, hi)`` of ``sorted_expert_ids``. gemm1's work list puts every
(m, n) tile of an expert together, n-major inside the expert so consecutive
blocks share the W13 slab::

    tile_map[NB_N*lo + n*(hi - lo) + (m - lo)] = m << 3 | n      for valid m, n < NB_N
    tile_map[valid_blocks*NB_N .. grid)        = -1               (idle blocks)
    tile_map[grid]                              = valid_blocks*NB_N

Each thread handles blocks ``tid, tid+256, ..``; ``lo`` / ``hi`` come from two
binary searches over the (non-decreasing) expert ids, branch-free.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr import range_constexpr

_THREADS = 256
_MAX_BLOCKS = 4096  # >= max_sorted / BM at 32768 tokens (1409)
_SEARCH_STEPS = 13  # 2^13 > _MAX_BLOCKS


def compile_tile_map(*, I: int, BM: int = 128):  # noqa: E741
    NB_N = I // 128
    BLOCK_ITERS = _MAX_BLOCKS // _THREADS
    TAIL_ITERS = (_MAX_BLOCKS * NB_N + 8) // _THREADS + 1

    @flyc.kernel
    def kernel_tile_map(
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tile_map: fx.Tensor,
        num_m_blocks: fx.Int32,
        grid_entries: fx.Int32,
    ):
        tid = fx.thread_idx.x
        nv_rsrc = buffer_ops.create_buffer_resource(
            num_valid_ids, max_size=False, num_records_bytes=4
        )
        num_valid = fx.Int32(
            buffer_ops.buffer_load(
                nv_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32, is_scalar=True
            )
        )
        vb = num_valid // fx.Int32(BM)  # valid blocks (num_valid is BM-padded)
        eid_rsrc = buffer_ops.create_buffer_resource(
            sorted_expert_ids, max_size=False, num_records_bytes=num_m_blocks * 4
        )
        tm_rsrc = buffer_ops.create_buffer_resource(
            tile_map, max_size=False, num_records_bytes=(grid_entries + 1) * 4
        )

        def _eid(i):
            return fx.Int32(
                buffer_ops.buffer_load(eid_rsrc, i, vec_width=1, dtype=fx.Int32)
            )

        def _bound(e, upper):
            """first index in [0, vb) whose expert id is > e (upper) / >= e (lower)"""
            lo = fx.Int32(0)
            n = vb
            for _ in range_constexpr(_SEARCH_STEPS):
                half = n // fx.Int32(2)
                mid = lo + half
                probe = _eid(_min(mid, vb - fx.Int32(1)))
                go_right = (n > fx.Int32(0)) & ((probe <= e) if upper else (probe < e))
                lo = fx.arith.select(go_right, mid + fx.Int32(1), lo)
                n = fx.arith.select(go_right, n - half - fx.Int32(1), half)
            return lo

        def _min(a, b):
            return fx.arith.select(a < b, a, b)

        for it in range_constexpr(BLOCK_ITERS):
            m = tid + fx.Int32(it * _THREADS)
            if m < vb:
                e = _eid(m)
                lo = _bound(e, False)
                hi = _bound(e, True)
                cnt = hi - lo
                base = lo * fx.Int32(NB_N) + (m - lo)
                for n in range_constexpr(NB_N):
                    buffer_ops.buffer_store(
                        (m << 3) | fx.Int32(n), tm_rsrc, base + cnt * fx.Int32(n)
                    )
        tail0 = vb * fx.Int32(NB_N)
        for it in range_constexpr(TAIL_ITERS):
            i = tail0 + tid + fx.Int32(it * _THREADS)
            if i < grid_entries:
                buffer_ops.buffer_store(fx.Int32(-1), tm_rsrc, i)
        if tid == fx.Int32(0):
            buffer_ops.buffer_store(tail0, tm_rsrc, grid_entries)

    @flyc.jit
    def launch_tile_map(
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tile_map: fx.Tensor,
        num_m_blocks: fx.Int32,
        grid_entries: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_tile_map(
            sorted_expert_ids, num_valid_ids, tile_map, num_m_blocks, grid_entries
        ).launch(grid=(1, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    return launch_tile_map


def tile_map_grid(num_m_blocks: int, I: int) -> int:  # noqa: E741
    """gemm1 grid for the table: every (m, n) tile of the allocation, rounded to 8
    XCDs"""
    return (num_m_blocks * (I // 128) + 7) // 8 * 8
