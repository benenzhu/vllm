# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL MiniMax-M3 indexer kernels for the fp8 (e4m3) index cache: drop-ins for
the AITER ``pa_sparse_block_score_prefill`` / ``pa_sparse_block_score_decode`` /
``pa_sparse_block_topk`` calls of ``indexer_aiter.py`` (same signatures, same score
buffer contract, same sentinels)."""

from .host import score_decode, score_prefill, topk

__all__ = ["score_decode", "score_prefill", "topk"]
