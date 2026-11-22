"""Cross-implementation parity tests for fused attention.

Layers under test:
  - torch reference (attention.attention_reference, naive fp32)
  - Triton fused forward (attention_triton.attention_triton)
  - torch SDPA (attention.attention_sdpa)

CUDA-only tests skip cleanly on CPU-only hosts.
"""

import pytest
import torch

from sifr_ml_systems.kernels.attention import attention_reference, attention_sdpa

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype,tolr", [(torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_matches_reference_cuda(dtype, tolr, causal) -> None:
    pytest.importorskip("triton")
    from sifr_ml_systems.kernels.attention_triton import attention_triton

    torch.manual_seed(0)
    q = torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype)
    v = torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype)

    ref = attention_reference(q, k, v, causal=causal)
    out = attention_triton(q, k, v, causal=causal)
    torch.testing.assert_close(out.float(), ref.float(), rtol=tolr, atol=tolr)


@pytest.mark.parametrize("dtype,tolr", [(torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_matches_sdpa_cuda(dtype, tolr, causal) -> None:
    pytest.importorskip("triton")
    from sifr_ml_systems.kernels.attention_triton import attention_triton

    torch.manual_seed(0)
    q = torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype)
    k = torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype)
    v = torch.randn(2, 4, 128, 64, device="cuda", dtype=dtype)

    ref = attention_sdpa(q, k, v, causal=causal)
    out = attention_triton(q, k, v, causal=causal)
    torch.testing.assert_close(out.float(), ref.float(), rtol=tolr, atol=tolr)


@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("seq_len", [256, 1024])
def test_triton_scales_cuda(head_dim, seq_len) -> None:
    pytest.importorskip("triton")
    from sifr_ml_systems.kernels.attention_triton import attention_triton

    torch.manual_seed(0)
    q = torch.randn(2, 2, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 2, seq_len, head_dim, device="cuda", dtype=torch.float16)

    ref = attention_reference(q, k, v)
    out = attention_triton(q, k, v)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_triton_rejects_mismatched_shapes_cuda() -> None:
    pytest.importorskip("triton")
    from sifr_ml_systems.kernels.attention_triton import attention_triton

    q = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 32, 64, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError):
        attention_triton(q, k, v)


def test_sdpa_matches_reference_cuda() -> None:
    q = torch.randn(2, 2, 64, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, 64, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 2, 64, 64, device="cuda", dtype=torch.float16)

    ref = attention_reference(q, k, v)
    out = attention_sdpa(q, k, v)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
