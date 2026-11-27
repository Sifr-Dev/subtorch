"""CPU correctness contract for the reference and SDPA attention paths.

`tests/test_attention_parity.py` is entirely GPU-gated, so on a CPU-only
runner nothing exercised `attention.attention_reference` or
`attention.attention_sdpa` at all. These tests run anywhere: they pin the
mathematical contract (softmax weighting, causal masking, dtype/shape
preservation, shape validation) that the Triton kernel is compared against,
so a regression in the reference would otherwise only surface as a confusing
GPU parity failure.
"""

import math

import pytest
import torch

from sifr_ml_systems.kernels.attention import attention_reference, attention_sdpa


def _randn(batch=2, heads=2, seq=8, head_dim=16, dtype=torch.float32):
    torch.manual_seed(0)
    shape = (batch, heads, seq, head_dim)
    return (
        torch.randn(shape, dtype=dtype),
        torch.randn(shape, dtype=dtype),
        torch.randn(shape, dtype=dtype),
    )


def test_reference_matches_explicit_softmax() -> None:
    q, k, v = _randn()
    out = attention_reference(q, k, v)

    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    expected = torch.einsum("bhqk,bhkd->bhqd", weights, v)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


def test_reference_rows_sum_to_one_after_weighting() -> None:
    """Each query attends with weights summing to 1, so a constant V returns itself."""
    q, k, _ = _randn()
    constant = torch.full((2, 2, 8, 16), 3.5)
    out = attention_reference(q, k, constant)
    torch.testing.assert_close(out, constant, rtol=1e-5, atol=1e-6)


def test_reference_causal_ignores_future_positions() -> None:
    """Changing a later key/value must not affect earlier query outputs."""
    q, k, v = _randn(seq=6)
    base = attention_reference(q, k, v, causal=True)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, -1, :] = 100.0
    v2[:, :, -1, :] = -100.0
    perturbed = attention_reference(q, k2, v2, causal=True)

    # Every position except the last has no visibility of the final token.
    torch.testing.assert_close(perturbed[:, :, :-1], base[:, :, :-1], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(perturbed[:, :, -1], base[:, :, -1])


def test_reference_rejects_mismatched_shapes() -> None:
    q, k, _ = _randn()
    bad = torch.randn(2, 2, 8, 32)
    with pytest.raises(ValueError, match="identical shapes"):
        attention_reference(q, k, bad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_reference_preserves_dtype_and_shape(dtype) -> None:
    q, k, v = _randn(dtype=dtype)
    out = attention_reference(q, k, v)
    assert out.dtype == dtype
    assert out.shape == q.shape


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_matches_reference_on_cpu(causal) -> None:
    q, k, v = _randn(dtype=torch.float32)
    ref = attention_reference(q, k, v, causal=causal)
    out = attention_sdpa(q, k, v, causal=causal)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


def test_sdpa_preserves_dtype_and_shape() -> None:
    q, k, v = _randn(dtype=torch.float16)
    out = attention_sdpa(q, k, v)
    assert out.dtype == torch.float16
    assert out.shape == q.shape
