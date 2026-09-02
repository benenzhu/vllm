# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMP dev harness for rocm_fp32_router_gemm_triton -- drop before the PR.

    python3 tests/kernels/moe/test_rocm_fp32_router_gemm.py

Correctness vs F.linear(x.float(), w) over M / shapes / strided-x, then a
steady-state bench that rotates input sets so the fp32 router weight is not
LLC-resident between launches -- in service each layer streams its own ~3MB
weight every step, so a hot-cache bench flatters the kernel. Sweeping past the
dispatch cutoff shows why it exists. Bench shape after FlyDSL's
test_fp4_gemm_4wave --vs-aiter.
"""

import torch

from vllm.model_executor.layers.fused_moe.router.rocm_fp32_router_gemm_triton import (  # noqa: E501
    ROCM_TRITON_ROUTER_MAX_TOKENS,
    _rocm_fp32_router_gemm,
    rocm_fp32_router_gemm_dispatch,
)


def ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(x.float(), w)


def check() -> None:
    torch.manual_seed(0)
    # (N, K): M3 router; M2.5 router; K not a multiple of BLOCK_K (k_mask)
    for n, k in [(128, 6144), (256, 3072), (96, 2000)]:
        w = torch.randn(n, k, device="cuda") / k**0.5
        for m in [1, 2, 3, 5, 8, 16, 17, 32, 64]:
            xs = torch.randn(m, k + 256, device="cuda", dtype=torch.bfloat16)
            for x in (xs[:, :k], xs[:, :k].contiguous()):  # strided + dense rows
                got, want = _rocm_fp32_router_gemm(x, w), ref(x, w)
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-3)
                # what the router actually consumes
                assert (got.topk(8, -1).indices == want.topk(8, -1).indices).all()
    # non-unit stride along K: the kernel cannot express it, dispatch must
    # route it to F.linear rather than read the wrong elements.
    x = torch.randn(8, 6144, 2, device="cuda", dtype=torch.bfloat16)[..., 0]
    w = torch.randn(128, 6144, device="cuda") / 6144**0.5
    torch.testing.assert_close(
        rocm_fp32_router_gemm_dispatch(x, w), ref(x, w), rtol=1e-4, atol=1e-3
    )
    print("correctness OK")


SET_BYTES = 512 << 20  # rotated working set > MI355X LLC
PAIRS, REPLAYS = 3, 8


def graph_us(fn, xs, ws) -> float:
    """CUDA-graph-replay timing: one launch per rotated set, replay the graph.

    Eager per-launch timing bottoms out at ~12us of Python launch overhead and
    cannot rank these kernels; graph replay amortizes it away and matches how
    vLLM actually runs decode (full cudagraph).
    """
    sets = len(xs)
    fn(xs[0], ws[0])
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for s in range(sets):
            fn(xs[s], ws[s])
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(PAIRS):
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(REPLAYS):
            g.replay()
        en.record()
        torch.cuda.synchronize()
        best = min(best, st.elapsed_time(en) / (REPLAYS * sets) * 1e3)
    return best


def bench(m: int, n: int = 128, k: int = 6144) -> None:
    sets = max(2, SET_BYTES // (n * k * 4 + m * k * 2))
    xs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(sets)]
    ws = [torch.randn(n, k, device="cuda") for _ in range(sets)]
    tri = graph_us(_rocm_fp32_router_gemm, xs, ws)
    base = graph_us(ref, xs, ws)
    used = "triton" if m <= ROCM_TRITON_ROUTER_MAX_TOKENS else "F.linear"
    print(
        f"M={m:5d} ({sets:3d} sets)  triton {tri:7.1f} us"
        f"   F.linear(fp32) {base:7.1f} us   dispatch -> {used}"
    )


if __name__ == "__main__":
    check()
    print(f"\nbench N=128 K=6144, cutoff M<={ROCM_TRITON_ROUTER_MAX_TOKENS}")
    for m in [1, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]:
        bench(m)
