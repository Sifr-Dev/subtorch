"""Kernel-level implementations and their reference semantics."""

from sifr_ml_systems.kernels.attention import attention_reference, attention_sdpa
from sifr_ml_systems.kernels.attention_triton import attention_triton, triton_attention_available
from sifr_ml_systems.kernels.rmsnorm import RMSNorm, rmsnorm
from sifr_ml_systems.kernels.rmsnorm_jax import jax_available, rmsnorm_jax, rmsnorm_jax_jit
from sifr_ml_systems.kernels.rmsnorm_torch import rmsnorm_compiled
from sifr_ml_systems.kernels.rmsnorm_triton import rmsnorm_triton, triton_available

__all__ = [
    "RMSNorm",
    "rmsnorm",
    "rmsnorm_triton",
    "triton_available",
    "rmsnorm_torch" ,
    "rmsnorm_compiled",
    "rmsnorm_jax",
    "rmsnorm_jax_jit",
    "jax_available",
    "attention_reference",
    "attention_sdpa",
    "attention_triton",
    "triton_attention_available",
]
