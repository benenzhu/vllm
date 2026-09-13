# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight shuffle, compile cache and launch wrapper for the skinny bf16 GEMM."""

import functools
import os

import flydsl.compiler as flyc
import torch

from .skinny import DEFAULT_TILE, MAX_RT, TILE_K, compile_skinny

MAX_TOKENS = MAX_RT * 16


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` bf16 (nn.Linear layout) -> 1 KB blocks of 16 columns x 32 K in the
    MFMA 16x16x32 B-fragment order: ``[N/16, K/32, klane 4, n 16, 8]``."""
    N, K = w.shape
    assert w.dtype == torch.bfloat16 and N % 16 == 0 and K % TILE_K == 0
    return w.view(N // 16, 16, K // TILE_K, 4, 8).permute(0, 2, 3, 1, 4).contiguous()


# (TILE_N, K-waves, prefetch, K-split workgroups) per (N, K): 256 CUs, the K-split
# spreads the per-CU activation ingest (09-13 sweeps, see skinny.py)
TILES = {
    (6144, 2048): (32, 4, 4, 1),  # o_proj: 192 strips (the K-split hand-off costs more than it saves)
    (2304, 6144): (48, 4, 4, 4),  # qkv: 48 strips x 4 = 192 workgroups (A per CU is 3x o_proj's)
}


def tile_for(n_tokens: int, N: int, K: int):
    """(TILE_N, K-waves, W prefetch, K-splits) by shape; ``M3_SKINNY_TILE=tn,kw,pf,ks``
    overrides for lab sweeps."""
    env = os.environ.get("M3_SKINNY_TILE")
    if env:
        return tuple(int(v) for v in env.split(","))
    return TILES.get((N, K), DEFAULT_TILE)


_scratch: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def scratch_for(launch, device) -> tuple[int, int]:
    """Device addresses of the K-split scratch (fp32 partials, per-strip arrival
    counters) for ``launch``; (0, 0) when the kernel does not split K. One buffer
    per kernel: the counters only grow (each launch adds exactly KS per strip), so
    they are never reset; launches of the same kernel must not overlap."""
    part_elems, cnt_elems = launch.scratch
    if part_elems == 0:
        return 0, 0
    key = (launch.kernel_name, str(device))
    if key not in _scratch:
        _scratch[key] = (
            torch.empty(part_elems, dtype=torch.float32, device=device),
            torch.zeros(cnt_elems, dtype=torch.int32, device=device),
        )
    part, cnt = _scratch[key]
    return part.data_ptr(), cnt.data_ptr()


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


_launches: dict[str, object] = {}


@functools.cache
def get_skinny(**kw):
    launch = compile_skinny(**kw)
    return _launches.setdefault(launch.kernel_name, launch)


def skinny_gemm(x: torch.Tensor, w_shuffled: torch.Tensor, N: int, K: int, out: torch.Tensor | None = None):
    """``x[M, K] bf16 @ w[N, K]^T`` -> bf16 ``[M, N]`` for M <= 128 (fp32 accumulation).
    ``w_shuffled`` is ``shuffle_weight(w)``."""
    M = x.shape[0]
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[1] == K
    assert 1 <= M <= MAX_TOKENS, M
    RT = (M + 15) // 16
    TILE_N, KW, pf, KS = tile_for(M, N, K)
    launch = get_skinny(N=N, K=K, RT=RT, TILE_N=TILE_N, KW=KW, prefetch=pf, KS=KS)
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    part, cnt = scratch_for(launch, x.device)
    _run_compiled(
        launch,
        x.data_ptr(),
        w_shuffled.data_ptr(),
        out.data_ptr(),
        int(M),
        part,
        cnt,
        torch.cuda.current_stream(),
    )
    return out
