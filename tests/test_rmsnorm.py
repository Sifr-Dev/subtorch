"""Correctness contract for the RMSNorm implementations (CPU-verifiable)."""

import pytest
import torch

from sifr_ml_systems.kernels.rmsnorm import rmsnorm


def test_rmsnorm_matches_manual_formula() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 64)
    weight = torch.randn(64).abs() + 0.5

    out = rmsnorm(x, weight, eps=1e-6)
    mean_square = x.float().pow(2).mean(-1, keepdim=True)
    expected = (x.float() * torch.rsqrt(mean_square + 1e-6) * weight.float()).to(x.dtype)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_weight_validation() -> None:
    with pytest.raises(ValueError):
        rmsnorm(torch.randn(2, 4), torch.ones(3))


def test_rmsnorm_preserves_dtype_and_shape() -> None:
    x = torch.randn(3, 2, 32, dtype=torch.bfloat16)
    weight = torch.ones(32, dtype=torch.float32)
    out = rmsnorm(x, weight)
    assert out.dtype == x.dtype
    assert out.shape == x.shape


def test_rmsnorm_is_finite_for_large_values() -> None:
    x = torch.full((4, 16), 1e4)
    out = rmsnorm(x, torch.ones(16))
    assert torch.isfinite(out).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_rmsnorm_triton_matches_reference() -> None:
    pytest.importorskip("triton")

    from sifr_ml_systems.kernels.rmsnorm_triton import rmsnorm_triton

    torch.manual_seed(0)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32).abs() + 0.5

    ref = rmsnorm(x, weight)
    out = rmsnorm_triton(x, weight)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-3, atol=2e-3)
