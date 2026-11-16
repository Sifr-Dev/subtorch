"""JAX reference for RMSNorm semantics.

Semantic parity contract with `rmsnorm.py` (torch) and `rmsnorm_triton.py`
(Triton): RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight, computed in
float32 internally regardless of input dtype.

JAX is an optional dependency (cuda-enabled builds preferred). Importing
this module does not require JAX; `jax_available()` reports whether a
usable CUDA-capable JAX is present, and calls raise a clear error
otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax
    import jax.numpy as jnp


def jax_available() -> bool:
    try:
        import jax
    except ImportError:
        return False
    try:
        devices = jax.devices()
    except Exception:
        return False
    return any("cuda" in device.platform.lower() or device.platform == "gpu" for device in devices)


def _rmsnorm_jax(x, weight, eps: float):
    import jax.numpy as jnp

    x32 = x.astype(jnp.float32)
    mean_square = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    import jax

    normalized = x32 * jax.lax.rsqrt(mean_square + eps)
    out = normalized * weight.astype(jnp.float32)
    return out.astype(x.dtype)


def rmsnorm_jax(x, weight, eps: float = 1e-6):
    """Un-jitted RMSNorm. `x` is a jax array; leading dims are flattened over the last axis."""
    try:
        import jax  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("jax is required for rmsnorm_jax; install cuda-enabled jax to compare on GPU") from exc
    return _rmsnorm_jax(x, weight, eps)


def rmsnorm_jax_jit(x, weight, eps: float = 1e-6):
    """Jit-compiled RMSNorm (same math, single XLA computation)."""
    try:
        import jax
    except ImportError as exc:
        raise RuntimeError("jax is required for rmsnorm_jax_jit; install cuda-enabled jax to compare on GPU") from exc
    return jax.jit(_rmsnorm_jax, static_argnums=(2,))(x, weight, eps)
