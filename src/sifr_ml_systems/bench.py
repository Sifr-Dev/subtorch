"""Minimal CUDA-event benchmark harness with warmup and medians."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch


def cuda_time(fn: Callable[[], torch.Tensor], warmup: int = 10, iterations: int = 100) -> dict:
    """Time a callable with CUDA events; falls back to wall clock on CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        elapsed_ms = []
        for _ in range(iterations):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
        elapsed_ms.sort()
        median = elapsed_ms[len(elapsed_ms) // 2]
        return {"device": "cuda", "warmup": warmup, "iterations": iterations, "median_ms": median}

    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total = (time.perf_counter() - start) * 1000
    return {"device": "cpu", "warmup": warmup, "iterations": iterations, "median_ms": total / iterations}
