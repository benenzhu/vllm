# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL skinny bf16 GEMM for the MiniMax-M3 decode projections (M <= 128)."""

from .host import MAX_TOKENS, shuffle_weight, skinny_gemm, supports, warmup

__all__ = ["MAX_TOKENS", "shuffle_weight", "skinny_gemm", "supports", "warmup"]
