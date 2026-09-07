# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode routing sort for the a16w4 MoE chain (M <= max_tokens): one launch.

Block 0 sorts the (token, slot) pairs by expert: every thread loads its <= PPT
pairs at once (one round trip), LDS histogram, a Hillis-Steele scan over the
padded counts, then placement through LDS cursors and padding; every other block
zeroes a slice of the output buffer that gemm2's atomics accumulate into. Same
output contract as aiter ``moe_sorting``:

  sorted_ids[row]        = token | slot << 24   (padding rows: token = n_tokens)
  sorted_weights[row]    = routing weight       (padding rows: 0)
  sorted_expert_ids[b]   = expert of block b (block_m rows)
  num_valid_ids[0]       = padded sorted row count
  out[n_tokens, H]       = 0 (bf16)

Rows inside one expert come out in the arrival order of the LDS atomics (not
deterministic, as in aiter); the gemms do not depend on it. aiter's
``moe_sorting`` is two kernels (~9.6 us together at M=64..256 in a graph).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr

from .utils import _lds_atomic_add_i32


def max_sorted_rows(n_tokens: int, E: int, topk: int, block_m: int) -> int:
    """aiter's bound: every active expert pads by < block_m rows."""
    active = min(E, n_tokens * topk)
    cumsum_max = n_tokens * topk + active * (block_m - 1)
    return ((cumsum_max + block_m - 1) // block_m) * block_m


threads = 256  # sorter block (one thread per expert, power of two for the scan)
zero_ctas = 127  # blocks that zero the output


@functools.cache
def compile_decode_sort(*, E: int, topk: int, block_m: int, H: int, max_tokens: int):
    assert (block_m & (block_m - 1)) == 0 and threads >= E and (H * 2) % 16 == 0
    bm_shift = block_m.bit_length() - 1
    PPT = (max_tokens * topk + threads - 1) // threads  # pairs per thread
    scan_rounds = threads.bit_length() - 1
    tag = f"E{E}_K{topk}_BM{block_m}_H{H}_M{max_tokens}_Z{zero_ctas}_T{threads}"

    @fx.struct
    class Shared:
        count: fx.Array[fx.Int32, E]  # rows per expert
        cursor: fx.Array[fx.Int32, E]  # next free sorted row per expert
        scan: fx.Array[fx.Int32, 2 * threads]  # ping-pong scan arrays

    @flyc.kernel(name=f"m3_decode_sort_zero_{tag}", known_block_size=[threads, 1, 1])
    def sort_zero(
        topk_ids: fx.Tensor,  # [n_tokens * topk] i32
        topk_w: fx.Tensor,  # [n_tokens * topk] f32
        sorted_ids: fx.Tensor,  # [max_sorted] i32
        sorted_w: fx.Tensor,  # [max_sorted] f32
        sorted_eids: fx.Tensor,  # [max_sorted / block_m] i32
        num_valid: fx.Tensor,  # [2] i32
        out_i32: fx.Tensor,  # out[n_tokens, H] bf16 viewed as i32 [n_tokens * H / 2]
        n_tok: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        if bx == 0:
            count, cursor = smem.count.ptr, smem.cursor.ptr
            scan = [smem.scan.ptr, smem.scan.ptr + threads]
            n_pairs = n_tok * topk
            last_pair = n_pairs - 1
            t_lt_E = tx < E

            # 0. this thread's pairs: one batch of loads (clamped index, validity)
            pairs = []
            for k in range_constexpr(PPT):
                p = tx + k * threads
                valid = p < n_pairs
                pc = valid.select(p, last_pair)
                pairs.append((p, valid, fx.Int32(topk_ids[pc]), fx.Float32(topk_w[pc])))

            # 1. zero the counters
            if t_lt_E:
                count[tx] = 0
            fx.gpu.barrier()

            # 2. histogram
            for p, valid, e, _w in pairs:
                if valid:
                    _lds_atomic_add_i32(count + e, 1)
            fx.gpu.barrier()

            # 3. inclusive Hillis-Steele scan of the padded counts (0 beyond E)
            cnt = t_lt_E.select(
                fx.Int32(count[t_lt_E.select(tx, fx.Int32(0))]), fx.Int32(0)
            )
            padded = (cnt + (block_m - 1)) & ~(block_m - 1)
            scan[0][tx] = padded
            fx.gpu.barrier()
            src, dst = 0, 1
            for r in range_constexpr(scan_rounds):
                d = 1 << r
                mine = fx.Int32(scan[src][tx])
                has = tx >= d
                other = fx.Int32(scan[src][has.select(tx - d, tx)])
                scan[dst][tx] = has.select(mine + other, mine)
                fx.gpu.barrier()
                src, dst = dst, src
            end = fx.Int32(scan[src][tx])
            start = end - padded

            # 4. thread e: cursor, expert ids of its blocks, padding rows, total
            if t_lt_E:
                cursor[tx] = start
                if tx == E - 1:
                    num_valid[0] = end
                for b in range(start >> bm_shift, end >> bm_shift):
                    sorted_eids[fx.Int32(b)] = tx
                for r in range(start + cnt, end):
                    sorted_ids[fx.Int32(r)] = n_tok  # padding: token n_tokens, slot 0
                    sorted_w[fx.Int32(r)] = fx.Float32(0.0)
            fx.gpu.barrier()

            # 5. place the pairs
            for p, valid, e, w in pairs:
                if valid:
                    row = _lds_atomic_add_i32(cursor + e, 1)
                    sorted_ids[row] = ((p // topk) & 0x00FFFFFF) | ((p % topk) << 24)
                    sorted_w[row] = w
        else:
            # blocks 1..zero_ctas: out[n_tokens, H] = 0, 16 B per store
            # 16 B per store through a (4, n/4) view (no vector-element pointer store
            # in this flydsl)
            out16 = fx.logical_divide(out_i32, fx.make_layout(4, 1))
            atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            zero_r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            zero_r.store(fx.Vector.filled(4, 0, fx.Int32))
            n_chunks = n_tok * (H * 2 // 16)
            for i in range((bx - 1) * threads + tx, n_chunks, zero_ctas * threads):
                fx.copy(atom, zero_r, fx.slice(out16, (None, fx.Int32(i))))

    @flyc.jit
    def launch(
        topk_ids: fx.Tensor,
        topk_w: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_w: fx.Tensor,
        sorted_eids: fx.Tensor,
        num_valid: fx.Tensor,
        out_i32: fx.Tensor,
        n_tokens: fx.Int32,
        stream: fx.Stream,
    ):
        sort_zero(
            topk_ids,
            topk_w,
            sorted_ids,
            sorted_w,
            sorted_eids,
            num_valid,
            out_i32,
            n_tokens,
        ).launch(grid=(1 + zero_ctas, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


def moe_sort_decode(topk_ids, topk_weights, E, H, block_m, out):
    """Drop-in for ``moe_sorting(topk_ids, topk_w, E, H, dtype, block_size)`` ->
    (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids); ``out``
    (``[n_tokens, H]`` bf16, contiguous) is zeroed in place."""
    n_tokens, topk = topk_ids.shape
    dev = topk_ids.device
    ms = max_sorted_rows(n_tokens, E, topk, block_m)
    sorted_ids = torch.empty(ms, dtype=torch.int32, device=dev)
    sorted_w = torch.empty(ms, dtype=torch.float32, device=dev)
    sorted_eids = torch.empty(ms // block_m, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    assert n_tokens <= 256
    launch = compile_decode_sort(
        E=E, topk=topk, block_m=block_m, H=H, max_tokens=64 if n_tokens <= 64 else 256
    )
    launch(
        topk_ids.contiguous().int().view(-1),
        topk_weights.contiguous().float().view(-1),
        sorted_ids,
        sorted_w,
        sorted_eids,
        num_valid,
        out.view(torch.int32).view(-1),
        int(n_tokens),
        torch.cuda.current_stream(),
    )
    return sorted_ids, sorted_w, sorted_eids, num_valid
