# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL MiniMax-M3 indexer kernels for the bf16 index cache on gfx950.

``host.index_score_prefill`` replaces the Triton prefill scorer
(``index_topk.minimax_m3_index_score``) for one local index head; the score
tensor layout is unchanged so ``minimax_m3_index_topk`` consumes it as is.
"""

from .host import index_decode, index_score_decode, index_score_prefill, index_topk_prefill

__all__ = ["index_decode", "index_score_decode", "index_score_prefill", "index_topk_prefill"]
