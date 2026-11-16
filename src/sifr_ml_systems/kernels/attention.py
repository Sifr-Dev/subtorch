"""Reference attention implementations.

`attention_reference` is the semantic source of truth: a naive fp32 softmax
(einsum QK^T → softmax → einsum with V). `attention_sdpa` wraps PyTorch's
fused SDPA as the strongest "free" baseline.

Causal masking via a lower-triangular boolean mask.
"""

from __future__ import annotations

import math

import torch


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Naive softmax attention in fp32. Shapes: (batch, heads, seq, head_dim)."""
    if q.shape != k.shape or k.shape != v.shape:
        msg = f"q, k, v must have identical shapes; got {q.shape}, {k.shape}, {v.shape}"
        raise ValueError(msg)

    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) / math.sqrt(q.shape[-1])
    if causal:
        seq_len = scores.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhqk,bhkd->bhqd", weights, vf)
    return out.to(q.dtype)


def attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """PyTorch SDPA (fused CUDA path when available)."""
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
