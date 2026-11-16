"""Fused attention forward kernel in Triton (FlashAttention-2 style).

One program per (query block, batch-head). Online softmax over key blocks:
running max, running denominator, and accumulator are carried in registers,
so the QK^T matrix is never materialized in HBM. Supports causal masking.

Semantics match `attention.attention_reference` (naive fp32 softmax) within
float16/bfloat16 tolerances; see tests/test_attention_parity.py.
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
    def _attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        seq_len,
        q_stride_bh,
        q_stride_m,
        k_stride_bh,
        k_stride_n,
        v_stride_bh,
        v_stride_n,
        out_stride_bh,
        out_stride_m,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        IEEE: tl.constexpr,
    ) -> None:
        block_m = tl.program_id(axis=0)
        off_bh = tl.program_id(axis=1)

        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        q_ptrs = q_ptr + off_bh * q_stride_bh + offs_m[:, None] * q_stride_m + offs_d[None, :]
        q = tl.load(q_ptrs)

        qk_scale: tl.constexpr = 1.0 / (HEAD_DIM ** 0.5)

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k_ptrs = k_ptr + off_bh * k_stride_bh + offs_n[:, None] * k_stride_n + offs_d[None, :]
            k = tl.load(k_ptrs)
            v_ptrs = v_ptr + off_bh * v_stride_bh + offs_n[:, None] * v_stride_n + offs_d[None, :]
            v = tl.load(v_ptrs)
            if IEEE:
                qk = tl.dot(q, tl.trans(k), input_precision="ieee") * qk_scale
            else:
                qk = tl.dot(q, tl.trans(k)).to(tl.float32) * qk_scale

            if CAUSAL:
                mask = offs_m[:, None] >= offs_n[None, :]
                qk = tl.where(mask, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            # Guard against a fully-masked first block in causal mode.
            m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            if IEEE:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, input_precision="ieee")
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        out_ptrs = out_ptr + off_bh * out_stride_bh + offs_m[:, None] * out_stride_m + offs_d[None, :]
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty))


def attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """Fused attention forward. q, k, v: (batch, heads, seq, head_dim), fp16/bf16/fp32."""
    if triton is None or _triton_import_error is not None:
        msg = f"Triton is required for attention_triton: {_triton_import_error}"
        raise RuntimeError(msg)

    if not q.is_cuda:
        msg = "attention_triton requires CUDA tensors"
        raise RuntimeError(msg)

    if q.shape != k.shape or k.shape != v.shape:
        msg = f"q, k, v must have identical shapes; got {q.shape}, {k.shape}, {v.shape}"
        raise ValueError(msg)

    batch, heads, seq_len, head_dim = q.shape
    if head_dim not in (16, 32, 64, 128):
        msg = f"head_dim must be one of 16/32/64/128; got {head_dim}"
        raise ValueError(msg)

    q_c = q.contiguous().view(batch * heads, seq_len, head_dim)
    k_c = k.contiguous().view(batch * heads, seq_len, head_dim)
    v_c = v.contiguous().view(batch * heads, seq_len, head_dim)

    out = torch.empty_like(q_c)
    block_m = 64 if head_dim <= 64 else 32
    block_n = 32

    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    _attention_forward_kernel[grid](
        q_c,
        k_c,
        v_c,
        out,
        seq_len,
        q_c.stride(0),
        q_c.stride(1),
        k_c.stride(0),
        k_c.stride(1),
        v_c.stride(0),
        v_c.stride(1),
        out.stride(0),
        out.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CAUSAL=causal,
        IEEE=(q.dtype == torch.float32),
    )
    return out.view(batch, heads, seq_len, head_dim)


def triton_attention_available() -> bool:
    return triton is not None and _triton_import_error is None
