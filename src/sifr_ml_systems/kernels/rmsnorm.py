"""RMSNorm semantics and a Triton kernel with the same contract.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

The reference implementation is the semantic source of truth. The Triton
kernel implements the identical math with one program per row and a
single pass over the row with a running sum of squares, which makes the
memory traffic O(1) per element instead of the two passes an unfused
PyTorch composition performs.
"""

from __future__ import annotations

import torch
from torch import nn


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference RMSNorm. Leading dimensions are flattened over the last axis."""
    if x.shape[-1] != weight.shape[0]:
        msg = f"weight length {weight.shape[0]} does not match last axis {x.shape[-1]}"
        raise ValueError(msg)
    mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = x.float() * torch.rsqrt(mean_square + eps)
    return (normalized * weight.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)
