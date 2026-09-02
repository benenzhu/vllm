# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Triton router GEMM: bf16 activations x fp32 router weight -> fp32 logits.

MoE routers (e.g. MiniMax-M3: hidden 6144, 128 experts, fp32 gate weight) hit
GateLinear's ultimate F.linear fallback on ROCm, which costs three kernels per
layer at decode:

    bf16->fp32 copy of x     (aten::to)
    hipBLASLt fp32 GEMM      (Cijk_...)
    split-K reduction        (Cijk_SS_..._PostGSU16)

For decode-sized M the GEMM is a skinny memory-bound read of the fp32 weight;
one Triton kernel does the whole thing with fp32 accumulation, so the result
keeps fp32-GEMM-grade accuracy (unlike downcasting the weight to bf16 to reach
the torch.mm out_dtype path). The grid splits BOTH output axes: one program per
(expert column, TILE_M-row tile). N alone is only 128 programs -- half of
MI355X's 256 CUs would sit idle; the M split multiplies that out so DRAM
latency on the weight rows overlaps across programs. Split-K was measured too
and loses badly: the cross-program reduction (fp32 atomics or a counter +
workspace handshake) serializes on a handful of cache lines and costs far more
than the parallelism buys.

This mirrors what the CUDA side already does for the same problem
(``csrc/libtorch_stable/fp32_router_gemm.cu``, dispatched by GateLinear tier 3):
a hand-written VALU dot product -- no matrix cores -- specialized to
decode-sized M, with large M handed back to the vendor GEMM.

Large M (prefill) falls back to F.linear, where the dense GEMM is the right
tool and these three kernels are noise. Without matrix cores, per-program work
grows linearly in M and the fp32 tile spills, so past the cutoff this kernel is
not merely unprofitable but far slower -- see the numbers on the cutoff below.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# Decode batches are tiny (tokens x spec factor); prefill goes to F.linear.
# MI355X N=128 K=6144, CUDA-graph-replay timed (eager per-launch timing bottoms
# out at ~12us of Python launch overhead and cannot rank these kernels; graph
# replay also matches how vLLM actually runs decode) with rotated inputs so the
# 3MB fp32 weight is never LLC-resident, triton vs F.linear in us:
#   M=1: 4.0 / 11.0   M=8: 4.5 / 11.5   M=16: 4.9 / 11.9
#   M=32: 6.3 / 14.1  M=64: 8.9 / 14.7  M=256: 24.2 / 18.7
# The crossover sits near M=192; 64 covers every decode batch this dispatch
# actually sees, so the cutoff stays there.
ROCM_TRITON_ROUTER_MAX_TOKENS = 64


@triton.jit
def _fp32_router_gemm_kernel(
    x_ptr,  # [M, K] bf16 (row-major, possibly strided rows)
    w_ptr,  # [N, K] fp32 (row-major, contiguous)
    out_ptr,  # [M, N] fp32 (row-major, contiguous)
    M,
    N,
    num_m_tiles,
    stride_xm,
    K: tl.constexpr,  # constexpr: unrolls the K loop and, when BLOCK_K
    TILE_M: tl.constexpr,  # divides K, compiles the k-masks out entirely
    BLOCK_K: tl.constexpr,
    XCD_REMAP: tl.constexpr,
):
    pid = tl.program_id(0)
    if XCD_REMAP:
        # MI3xx dispatches workgroups round-robin across the 8 XCDs, so
        # consecutive pids land on different dies. Remap so each XCD gets a
        # contiguous logical range: the M-tiles of one expert (which re-read
        # the same weight row) and neighboring experts (adjacent rows) then
        # share that die's L2. Bijective when the grid divides by 8; the
        # launcher only sets XCD_REMAP then.
        per_xcd = (N * num_m_tiles) // 8
        pid = (pid % 8) * per_xcd + pid // 8
    n = pid // num_m_tiles
    mb = pid % num_m_tiles
    offs_m = mb * TILE_M + tl.arange(0, TILE_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    EVEN_K: tl.constexpr = K % BLOCK_K == 0
    acc = tl.zeros([TILE_M], dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_K):
        if EVEN_K:
            w = tl.load(w_ptr + n * K + k0 + offs_k)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :],
                mask=m_mask[:, None],
                other=0.0,
            ).to(tl.float32)
        else:
            k_mask = k0 + offs_k < K
            w = tl.load(w_ptr + n * K + k0 + offs_k, mask=k_mask, other=0.0)
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + (k0 + offs_k)[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
        acc += tl.sum(x * w[None, :], axis=1)

    tl.store(out_ptr + offs_m * N + n, acc, mask=m_mask)


def _rocm_fp32_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=torch.float32, device=x.device)
    # MI355X sweep winners (graph-replay timed, rotated inputs). TILE_M grows
    # with the batch so the grid stays large without re-reading the weight row
    # too many times per expert.
    bm = max(triton.next_power_of_2(M), 1)
    if bm <= 4:
        tile_m, block_k, warps = 1, 2048, 4
    elif bm <= 16:
        tile_m, block_k, warps = 2, 2048, 4
    elif bm <= 32:
        tile_m, block_k, warps = 4, 2048, 2
    else:
        tile_m, block_k, warps = 8, 1024, 4
    num_m_tiles = triton.cdiv(M, tile_m)
    grid = N * num_m_tiles
    _fp32_router_gemm_kernel[(grid,)](
        x,
        weight,
        out,
        M,
        N,
        num_m_tiles,
        x.stride(0),
        K=K,
        TILE_M=tile_m,
        BLOCK_K=block_k,
        XCD_REMAP=int(grid % 8 == 0),
        num_warps=warps,
    )
    return out


def _rocm_fp32_router_gemm_dispatch_impl(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    # The kernel indexes x as offs_m * stride_xm + k, i.e. it handles a row
    # stride but assumes unit stride along K.
    if x.shape[0] <= ROCM_TRITON_ROUTER_MAX_TOKENS and x.stride(1) == 1:
        return _rocm_fp32_router_gemm(x, weight)
    return torch.nn.functional.linear(x.float(), weight)


def _rocm_fp32_router_gemm_dispatch_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.float32, device=x.device
    )


# Wrapped in a custom op for the same reason as fp32_router_gemm_dispatch:
# the num_tokens branch must not be frozen by torch.compile.
direct_register_custom_op(
    op_name="rocm_fp32_router_gemm_dispatch",
    op_func=_rocm_fp32_router_gemm_dispatch_impl,
    fake_impl=_rocm_fp32_router_gemm_dispatch_fake,
    mutates_args=[],
)


def rocm_fp32_router_gemm_dispatch(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return torch.ops.vllm.rocm_fp32_router_gemm_dispatch(x, weight)
