"""Fused RMSNorm forward kernel in Triton.

One program per row. The reference `rmsnorm` in `rmsnorm.py` defines the
semantics; this kernel must match it within float32 tolerance and is
covered by the same correctness tests.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError as exc:  # pragma: no cover - environment without triton
    triton = None

    class _MissingTriton:
        pass

    tl = _MissingTriton()
    _triton_import_error: Exception | None = exc
else:
    _triton_import_error = None


if triton is not None:

    @triton.jit
    def _rmsnorm_forward_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ) -> None:
        row = tl.program_id(axis=0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols

        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(x * x, axis=0) / n_cols
        inverse = tl.rsqrt(mean_square + eps)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        result = x * inverse * weight
        tl.store(out_ptr + row * n_cols + cols, result.to(out_ptr.dtype.element_ty), mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Fused RMSNorm forward pass. Requires CUDA and Triton."""
    if triton is None or _triton_import_error is not None:
        msg = f"Triton is required for rmsnorm_triton: {_triton_import_error}"
        raise RuntimeError(msg)

    if not x.is_cuda:
        msg = "rmsnorm_triton requires a CUDA tensor"
        raise RuntimeError(msg)

    contiguous = x.contiguous()
    if contiguous.shape[-1] != weight.shape[0]:
        msg = f"weight length {weight.shape[0]} does not match last axis {contiguous.shape[-1]}"
        raise ValueError(msg)

    n_cols = contiguous.shape[-1]
    rows = contiguous.numel() // n_cols
    block = triton.next_power_of_2(n_cols)

    out = torch.empty_like(contiguous)
    _rmsnorm_forward_kernel[(rows,)](contiguous, weight.contiguous(), out, n_cols, eps, BLOCK=block)
    return out


def triton_available() -> bool:
    return triton is not None and _triton_import_error is None
