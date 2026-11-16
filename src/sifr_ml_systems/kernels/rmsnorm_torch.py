"""torch.compile'd RMSNorm baseline.

Same math as `rmsnorm.rmsnorm` — useful as the strongest "free" baseline:
whatever Inductor/Triton autogeneration can do without a hand-written
kernel. Compilation happens on first call; callers should count
compilation inside their warm-up, not their timed loop.
"""

from __future__ import annotations

import torch

from sifr_ml_systems.kernels.rmsnorm import rmsnorm

_compiled_cache: dict[float, torch.nn.Module] = {}


def rmsnorm_compiled(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm via torch.compile (cached per eps value)."""
    fn = _compiled_cache.get(eps)
    if fn is None:
        fn = torch.compile(lambda a, b: rmsnorm(a, b, eps))  # type: ignore[assignment]
        _compiled_cache[eps] = fn
    return fn(x, weight)
