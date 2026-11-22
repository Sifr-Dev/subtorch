"""Reproducible per-iteration timing, memory, and numerical error metrics."""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch


def environment_info(device: torch.device) -> dict[str, Any]:
    """Identify the measured software stack and hardware, not a presumed GPU."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    from importlib.metadata import PackageNotFoundError, version

    for name in ("triton", "jax", "numpy"):
        try:
            info[name] = version(name)
        except PackageNotFoundError:
            info[name] = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            msg = "A CUDA benchmark was requested, but torch.cuda.is_available() is false"
            raise RuntimeError(msg)
        properties = torch.cuda.get_device_properties(device)
        info["hardware"] = {
            "name": properties.name,
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "multi_processor_count": properties.multi_processor_count,
        }
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            info["hardware"]["driver"] = driver.stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            info["hardware"]["driver_error"] = str(exc)
    return info


def write_json(result: dict[str, Any], output: str | Path | None) -> None:
    """Write complete JSON to a durable path; stdout remains machine-readable."""
    result["schema_version"] = 1
    result["recorded_at_utc"] = datetime.now(UTC).isoformat()
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        result["raw_json_path"] = str(path.resolve())
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


# Error metrics stream the result through bounded host chunks so that a
# benchmark device whose memory is reserved (e.g. JAX) still allows exact
# full-tensor error accounting. Tests shrink this constant to force every
# chunking edge (partial first chunk, many chunks, exact alignment).
HOST_ERROR_CHUNK_ELEMENTS = 1 << 20


def error_metrics(out: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float) -> dict[str, Any]:
    """Return measured fp32 errors and the exact torch.testing error bound."""
    if out.shape != expected.shape or out.dtype != expected.dtype:
        raise AssertionError(f"shape/dtype mismatch: {out.shape}/{out.dtype} vs {expected.shape}/{expected.dtype}")
    # JAX may reserve most of the GPU after timing. Compare bounded host chunks
    # instead of allocating several full fp32 tensors on the benchmark device.
    out_flat = out.detach().float().cpu().flatten()
    expected_flat = expected.detach().float().cpu().flatten()
    max_absolute = max_relative = 0.0
    absolute_sum = 0.0
    mismatches = 0
    max_excess = float("-inf")
    for start in range(0, out_flat.numel(), HOST_ERROR_CHUNK_ELEMENTS):
        actual = out_flat[start : start + HOST_ERROR_CHUNK_ELEMENTS]
        expected_chunk = expected_flat[start : start + HOST_ERROR_CHUNK_ELEMENTS]
        difference = (actual - expected_chunk).abs()
        bound = atol + rtol * expected_chunk.abs()
        torch.testing.assert_close(actual, expected_chunk, rtol=rtol, atol=atol)
        nonzero = expected_chunk != 0
        if nonzero.any():
            max_relative = max(max_relative, (difference[nonzero] / expected_chunk[nonzero].abs()).max().item())
        max_absolute = max(max_absolute, difference.max().item())
        absolute_sum += difference.sum().item()
        max_excess = max(max_excess, (difference - bound).max().item())
        mismatches += int((difference > bound).sum().item())
    mean_absolute = absolute_sum / out.numel() if out.numel() else 0.0
    if out.numel() == 0:
        max_excess = -atol
    return {
        "rtol": rtol,
        "atol": atol,
        "passed": True,
        "max_absolute_error": max_absolute,
        "mean_absolute_error": mean_absolute,
        "max_relative_error_nonzero": max_relative,
        "max_bound_excess": max_excess,
        "mismatched_elements": mismatches,
        "num_elements": out.numel(),
    }


def cuda_time(
    fn: Callable[[], Any],
    warmup: int = 10,
    iterations: int = 100,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Time each call, synchronizing around events; retain samples and peaks."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requested without an available CUDA device")
    is_cuda = device.type == "cuda"
    synchronize = torch.cuda.synchronize if is_cuda else (lambda: None)
    synchronize()
    for _ in range(warmup):
        fn()
    synchronize()
    memory: dict[str, int] = {}
    if is_cuda:
        memory["allocated_before_timed_bytes"] = torch.cuda.memory_allocated(device)
        memory["reserved_before_timed_bytes"] = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(iterations):
        if is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
    if is_cuda:
        memory.update(
            allocated_after_timed_bytes=torch.cuda.memory_allocated(device),
            reserved_after_timed_bytes=torch.cuda.memory_reserved(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    ordered = sorted(samples)
    result: dict[str, Any] = {
        "device": str(device),
        "timing": "cuda_events_per_iteration" if is_cuda else "perf_counter_per_iteration",
        "warmup": warmup,
        "iterations": iterations,
        "min_ms": ordered[0],
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "max_ms": ordered[-1],
        "sample_stdev_ms": statistics.stdev(samples) if iterations > 1 else 0.0,
        "p05_ms": quantile(ordered, 0.05),
        "p95_ms": quantile(ordered, 0.95),
        "samples_ms": samples,
    }
    if memory:
        result["cuda_memory"] = memory
    return result


def quantile(ordered: list[float], probability: float) -> float:
    """Linearly interpolate an order statistic; the input must be sorted."""
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[min(lower + 1, len(ordered) - 1)] * fraction


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU benchmark; no results were measured")
    return torch.device("cuda")


__all__ = ["cuda_time", "environment_info", "error_metrics", "quantile", "require_cuda", "write_json"]
