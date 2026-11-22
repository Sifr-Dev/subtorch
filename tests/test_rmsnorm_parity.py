"""Cross-implementation parity tests for RMSNorm.

Layers under test:
  - torch reference (rmsnorm.rmsnorm)
  - Triton fused kernel (rmsnorm_triton.rmsnorm_triton)
  - torch.compile baseline (rmsnorm_torch.rmsnorm_compiled)
  - JAX reference, un-jitted and jitted (rmsnorm_jax.rmsnorm_jax[_jit])

CUDA-only tests skip cleanly on CPU-only hosts. JAX tests additionally
require a CUDA-capable JAX install.
"""

import pytest
import torch

from sifr_ml_systems.kernels.rmsnorm import rmsnorm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _reference(x, weight, eps=1e-6):
    return rmsnorm(x, weight, eps)


@pytest.mark.parametrize("dtype,tolr", [(torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 2e-2)])
def test_triton_matches_reference_cuda(dtype, tolr) -> None:
    triton = pytest.importorskip("triton")
    del triton
    from sifr_ml_systems.kernels.rmsnorm_triton import rmsnorm_triton

    torch.manual_seed(0)
    x = torch.randn(256, 1024, device="cuda", dtype=dtype)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32).abs() + 0.5

    ref = _reference(x, weight)
    out = rmsnorm_triton(x, weight)
    torch.testing.assert_close(out.float(), ref.float(), rtol=tolr, atol=tolr)


@pytest.mark.parametrize("dtype,tolr", [(torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 2e-2)])
def test_torch_compile_matches_reference_cuda(dtype, tolr) -> None:
    from sifr_ml_systems.kernels.rmsnorm_torch import rmsnorm_compiled

    torch.manual_seed(0)
    x = torch.randn(256, 1024, device="cuda", dtype=dtype)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32).abs() + 0.5

    ref = _reference(x, weight)
    out = rmsnorm_compiled(x, weight)
    torch.testing.assert_close(out.float(), ref.float(), rtol=tolr, atol=tolr)


def _to_jax(tensor):
    jax = pytest.importorskip("jax")
    from jax.dlpack import from_dlpack

    return from_dlpack(tensor.contiguous())


@pytest.mark.parametrize("dtype,tolr", [(torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 2e-2)])
def test_jax_unjitted_matches_reference_cuda(dtype, tolr) -> None:
    jax = pytest.importorskip("jax")
    from sifr_ml_systems.kernels.rmsnorm_jax import jax_available

    if not jax_available():
        pytest.skip("JAX has no CUDA device in this environment")

    from sifr_ml_systems.kernels.rmsnorm_jax import rmsnorm_jax

    torch.manual_seed(0)
    x = torch.randn(256, 1024, device="cuda", dtype=dtype)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32).abs() + 0.5

    ref = _reference(x, weight)
    x_jax = _to_jax(x)
    w_jax = _to_jax(weight)
    out = rmsnorm_jax(x_jax, w_jax)
    out_torch = torch.from_dlpack(out.__dlpack__())
    torch.testing.assert_close(out_torch.float(), ref.float(), rtol=tolr, atol=tolr)


@pytest.mark.parametrize("dtype,tolr", [(torch.float32, 1e-5), (torch.float16, 2e-2), (torch.bfloat16, 2e-2)])
def test_jax_jitted_matches_reference_cuda(dtype, tolr) -> None:
    jax = pytest.importorskip("jax")
    from sifr_ml_systems.kernels.rmsnorm_jax import jax_available

    if not jax_available():
        pytest.skip("JAX has no CUDA device in this environment")

    from sifr_ml_systems.kernels.rmsnorm_jax import rmsnorm_jax_jit

    torch.manual_seed(0)
    x = torch.randn(256, 1024, device="cuda", dtype=dtype)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32).abs() + 0.5

    ref = _reference(x, weight)
    x_jax = _to_jax(x)
    w_jax = _to_jax(weight)
    out = rmsnorm_jax_jit(x_jax, w_jax)
    out_torch = torch.from_dlpack(out.__dlpack__())
    torch.testing.assert_close(out_torch.float(), ref.float(), rtol=tolr, atol=tolr)


def test_triton_and_jax_agree_with_each_other_cuda() -> None:
    pytest.importorskip("triton")
    jax = pytest.importorskip("jax")
    from sifr_ml_systems.kernels.rmsnorm_jax import jax_available, rmsnorm_jax_jit

    if not jax_available():
        pytest.skip("JAX has no CUDA device in this environment")
    from sifr_ml_systems.kernels.rmsnorm_triton import rmsnorm_triton

    torch.manual_seed(0)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.float16)
    weight = torch.randn(1024, device="cuda", dtype=torch.float16).abs() + 0.5

    out_triton = rmsnorm_triton(x, weight)
    out_jax = rmsnorm_jax_jit(_to_jax(x), _to_jax(weight))
    out_jax_torch = torch.from_dlpack(out_jax.__dlpack__())
    torch.testing.assert_close(out_triton.float(), out_jax_torch.float(), rtol=2e-2, atol=2e-2)
