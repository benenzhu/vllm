# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlyDSL skinny bf16 GEMM (decode projections) against torch."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and torch.cuda.is_available()),
    reason="ROCm FlyDSL kernel",
)


@pytest.mark.parametrize("M", [1, 4, 16, 17, 33, 80, 128])
@pytest.mark.parametrize("N,K", [(6144, 2048), (2304, 6144)])
def test_skinny_gemm(M, N, K):
    from vllm.models.minimax_m3.amd.ops.gemm_a16w16 import shuffle_weight, skinny_gemm

    torch.manual_seed(M * 7 + N)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.02
    ref = (x.float() @ w.float().T).to(torch.bfloat16)
    out = skinny_gemm(x, shuffle_weight(w), N, K)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=1e-2)
