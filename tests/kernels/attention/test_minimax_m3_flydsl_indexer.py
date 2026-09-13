# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL fp8 indexer kernels (gfx950,
``vllm/models/minimax_m3/amd/ops/index_fp8``) against the AITER kernels they
replace on the fp8 index-cache path: the prefill and decode scorers must be
bit-identical (same MFMA, exact max, same sentinels), the top-k must select the
same blocks and emit the same page table (a tie in the score may order two
blocks differently)."""

import random

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]

aiter_msa = pytest.importorskip("aiter.ops.msa_attention")
index_fp8 = pytest.importorskip("vllm.models.minimax_m3.amd.ops.index_fp8")

BLK = 128
D = 128
TOPK = 16
PAGES = 8
FP8 = torch.float8_e4m3fn
DEV = "cuda"


def _width(max_seq: int) -> int:
    """AITER's score-row width: a power of two of 64-block strips."""
    mb = -(-max_seq // BLK)
    return (1 << (-(-mb // 64) - 1).bit_length()) * 64


def _i32(x):
    return torch.tensor(x, dtype=torch.int32, device=DEV)


def _inputs(q_lens, ctxs, seed):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    seq_lens = [c + q for c, q in zip(ctxs, q_lens)]
    nblocks = [-(-s // BLK) for s in seq_lens]
    pool = sum(nblocks) + 8
    cache = (torch.randn(pool, BLK, D, device=DEV, generator=g) * 0.25).to(FP8)
    bt = torch.zeros(len(q_lens), max(nblocks) + 2, dtype=torch.int32, device=DEV)
    perm = torch.randperm(pool, device=DEV, generator=g).to(torch.int32)
    o = 0
    for i, n in enumerate(nblocks):
        bt[i, :n] = perm[o : o + n]
        o += n
    total_q = sum(q_lens)
    q = (torch.randn(total_q, 1, D, device=DEV, generator=g) * 0.25).to(FP8)
    score = torch.full((1, total_q, _width(max(seq_lens))), float("nan"), device=DEV)
    return cache, bt, _i32(seq_lens), q, score, seq_lens


PREFILL_CASES = [
    ([7], [0]),
    ([300], [1000]),
    ([512, 513, 100], [4000, 130000, 129]),
    ([1, 1029], [70000, 3]),
]


@pytest.mark.parametrize("q_lens,ctxs", PREFILL_CASES)
def test_prefill_scorer_matches_aiter(q_lens, ctxs):
    cache, bt, seq, q, ref, seq_lens = _inputs(q_lens, ctxs, seed=len(ctxs))
    out = ref.clone()
    cu = _i32([0] + list(torch.cumsum(torch.tensor(q_lens), 0)))
    kw = dict(
        init_blocks=0,
        local_blocks=1,
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
    )
    aiter_msa.pa_sparse_block_score_prefill(q, cache, ref, bt, cu, seq, **kw)
    index_fp8.score_prefill(q, cache, out, bt, cu, seq, **kw)
    torch.cuda.synchronize()
    for b, (ql, c) in enumerate(zip(q_lens, ctxs)):
        r0 = int(cu[b])
        for r in range(ql):
            n = (c + r) // BLK + 1  # the row's causal blocks
            assert torch.equal(out[0, r0 + r, :n], ref[0, r0 + r, :n]), (b, r)


DECODE_CASES = [
    ([900], 1),
    ([3000, 130000, 129, 70001], 4),
    ([5000, 6000], 16),
    (
        [random.Random(3).randint(1, 20000) for _ in range(70)],
        2,
    ),  # two 64-request groups
]
# the four-waves-per-row top-k: several chunks per wave, a wave with none
DECODE_TOPK_CASES = DECODE_CASES[:3] + [([600000, 350000, 1500], 4)]


@pytest.mark.parametrize("seq_lens,qlen", DECODE_CASES)
def test_decode_scorer_matches_aiter(seq_lens, qlen):
    cache, bt, seq, q, ref, _ = _inputs(
        [qlen] * len(seq_lens), [s - qlen for s in seq_lens], seed=qlen
    )
    out = ref.clone()
    kw = dict(init_blocks=0, local_blocks=1, query_len=qlen, max_seq_len=max(seq_lens))
    aiter_msa.pa_sparse_block_score_decode(q, cache, ref, bt, seq, **kw)
    index_fp8.score_decode(q, cache, out, bt, seq, **kw)
    torch.cuda.synchronize()
    for r in range(len(seq_lens)):
        for t in range(qlen):
            n = -(-(seq_lens[r] - qlen + t + 1) // BLK)
            assert torch.equal(out[0, r * qlen + t, :n], ref[0, r * qlen + t, :n]), (
                r,
                t,
            )


def _topk_both(score, bt, seq, total_q, max_seq, kw):
    outs = []
    for fn in (aiter_msa.pa_sparse_block_topk, index_fp8.topk):
        idx = torch.full((1, total_q, TOPK), -7, dtype=torch.int32, device=DEV)
        sbt = torch.full((total_q, TOPK * PAGES), -7, dtype=torch.int32, device=DEV)
        sctx = torch.full((total_q,), -7, dtype=torch.int32, device=DEV)
        fn(
            score,
            idx,
            bt,
            seq,
            sbt,
            sctx,
            max_seq_len=max_seq,
            block_size=BLK,
            num_kv_heads=1,
            pages_per_block=PAGES,
            **kw,
        )
        torch.cuda.synchronize()
        outs.append((idx, sbt, sctx))
    return outs


def _check_topk(score, ref, out):
    """Same selection unless the differing blocks tie in score; the table and
    token count must match wherever the selection does."""
    (i_ref, s_ref, c_ref), (i_out, s_out, c_out) = ref, out
    differ = (i_ref != i_out).any(dim=-1)[0]

    def sel_scores(idx, r):
        i = idx[0, r].clamp(min=0).long()
        s = torch.where(
            idx[0, r] >= 0, score[0, r, i], torch.full((TOPK,), -1e38, device=DEV)
        )
        return s.sort(descending=True).values

    for r in differ.nonzero().flatten().tolist():
        assert torch.equal(sel_scores(i_ref, r), sel_scores(i_out, r)), r
    same = ~differ
    assert torch.equal(s_ref[same], s_out[same])
    assert torch.equal(c_ref[same], c_out[same])


@pytest.mark.parametrize("q_lens,ctxs", PREFILL_CASES)
def test_prefill_topk_matches_aiter(q_lens, ctxs):
    cache, bt, seq, q, score, seq_lens = _inputs(q_lens, ctxs, seed=len(ctxs) + 11)
    cu = _i32([0] + list(torch.cumsum(torch.tensor(q_lens), 0)))
    aiter_msa.pa_sparse_block_score_prefill(
        q,
        cache,
        score,
        bt,
        cu,
        seq,
        init_blocks=0,
        local_blocks=1,
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
    )
    pos = [c + r for c, ql in zip(ctxs, q_lens) for r in range(ql)]
    kw = dict(
        num_valid_pages=_i32([p // BLK + 1 for p in pos]),
        kv_lens=_i32([p + 1 for p in pos]),
        row_req_id=_i32([b for b, ql in enumerate(q_lens) for _ in range(ql)]),
    )
    ref, out = _topk_both(score, bt, seq, sum(q_lens), max(seq_lens), kw)
    _check_topk(score, ref, out)


@pytest.mark.parametrize("seq_lens,qlen", DECODE_TOPK_CASES)
def test_decode_topk_matches_aiter(seq_lens, qlen):
    cache, bt, seq, q, score, _ = _inputs(
        [qlen] * len(seq_lens), [s - qlen for s in seq_lens], seed=qlen + 5
    )
    aiter_msa.pa_sparse_block_score_decode(
        q,
        cache,
        score,
        bt,
        seq,
        init_blocks=0,
        local_blocks=1,
        query_len=qlen,
        max_seq_len=max(seq_lens),
    )
    ref, out = _topk_both(
        score, bt, seq, len(seq_lens) * qlen, max(seq_lens), dict(query_len=qlen)
    )
    _check_topk(score, ref, out)


# ---------------------------------------------------------------------------
# bf16 index cache: the FlyDSL drop-ins of the Triton wrappers in amd/ops/index_topk
# ---------------------------------------------------------------------------
index_bf16 = pytest.importorskip("vllm.models.minimax_m3.amd.ops.index_bf16")


def _bf16_inputs(q_lens, ctxs, seed):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    seq_lens = [c + q for c, q in zip(ctxs, q_lens)]
    nblocks = [-(-s // BLK) for s in seq_lens]
    pool = sum(nblocks) + 8
    cache = (torch.randn(pool, BLK, D, device=DEV, generator=g) * 0.1).to(
        torch.bfloat16
    )
    bt = torch.zeros(len(q_lens), max(nblocks), dtype=torch.int32, device=DEV)
    perm = torch.randperm(pool, device=DEV, generator=g).to(torch.int32)
    o = 0
    for i, n in enumerate(nblocks):
        bt[i, :n] = perm[o : o + n]
        o += n
    q = (torch.randn(sum(q_lens), 1, D, device=DEV, generator=g) * 0.1).to(
        torch.bfloat16
    )
    return cache, bt, q, _i32(seq_lens), _i32(ctxs), seq_lens


@pytest.mark.parametrize("q_lens,ctxs", PREFILL_CASES)
def test_bf16_prefill_matches_triton(q_lens, ctxs):
    from vllm.models.minimax_m3.amd.ops import index_topk as tri

    cache, bt, q, seq, prefix, seq_lens = _bf16_inputs(q_lens, ctxs, seed=len(ctxs) + 3)
    cu = _i32([0] + list(torch.cumsum(torch.tensor(q_lens), 0)))
    args = (q, cache, bt, cu, seq, prefix, max(q_lens), max(seq_lens), 1)
    # the Triton kernels through the wrappers with the FlyDSL dispatch off
    tri._flydsl_bf16.cache_clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tri.envs, "VLLM_ROCM_USE_M3_FLYDSL_INDEXER", False)
        ref = tri.minimax_m3_index_score(*args)
        tk_ref = tri.minimax_m3_index_topk(ref, cu, prefix, max(q_lens), TOPK, 0, 1)
    tri._flydsl_bf16.cache_clear()
    out = index_bf16.index_score_prefill(*args)
    tk_out = index_bf16.index_topk_prefill(out, cu, prefix, max(q_lens), TOPK, 0, 1)
    torch.cuda.synchronize()
    for b, (ql, c) in enumerate(zip(q_lens, ctxs)):
        r0 = int(cu[b])
        for r in range(ql):
            n = (c + r) // BLK + 1
            torch.testing.assert_close(
                out[0, r0 + r, :n], ref[0, r0 + r, :n], rtol=0, atol=1e-5
            )
    # the same block set per row (ties in the fp32 sum order may reorder)
    for r in range(sum(q_lens)):
        assert set(tk_out[0, r].tolist()) == set(tk_ref[0, r].tolist()), r


@pytest.mark.parametrize("seq_lens,qlen", DECODE_CASES)
def test_bf16_decode_matches_triton(seq_lens, qlen):
    from vllm.models.minimax_m3.amd.ops import index_topk as tri
    from vllm.models.minimax_m3.amd.ops.sparse_pa import PAGES_PER_SPARSE_BLOCK

    cache, bt, q, seq, _, _ = _bf16_inputs(
        [qlen] * len(seq_lens), [s - qlen for s in seq_lens], seed=qlen + 9
    )
    total_q = q.shape[0]
    args = (q, cache, bt, seq, max(seq_lens), TOPK, 0, 1, 1, qlen, qlen)

    def run(fn):
        out = torch.empty(1, total_q, TOPK, dtype=torch.int32, device=DEV)
        sbt = torch.empty(
            total_q, TOPK * PAGES_PER_SPARSE_BLOCK, dtype=torch.int32, device=DEV
        )
        sctx = torch.empty(total_q, dtype=torch.int32, device=DEV)
        cnt = torch.zeros(1, total_q, dtype=torch.int32, device=DEV)
        fn(
            *args,
            out=out,
            attention_block_table=bt,
            sparse_block_table_out=sbt,
            sparse_context_lens_out=sctx,
            block_page_stride=PAGES_PER_SPARSE_BLOCK,
            completion_counter=cnt,
        )
        torch.cuda.synchronize()
        return out, sbt, sctx

    tri._flydsl_bf16.cache_clear()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tri.envs, "VLLM_ROCM_USE_M3_FLYDSL_INDEXER", False)
        o_ref, s_ref, c_ref = run(tri.minimax_m3_index_decode)
    tri._flydsl_bf16.cache_clear()
    o_out, s_out, c_out = run(index_bf16.index_decode)
    for r in range(total_q):
        assert set(o_out[0, r].tolist()) == set(o_ref[0, r].tolist()), r
    assert torch.equal(c_ref, c_out)
    same = ~(o_ref != o_out).any(dim=-1)[0]
    assert torch.equal(s_ref[same], s_out[same])
