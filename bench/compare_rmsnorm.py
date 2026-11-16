"""Reproducible RMSNorm benchmark: reference, Triton, torch.compile, JAX.

Usage:
    python -m bench.compare_rmsnorm --rows 4096 --cols 4096 --dtype float16
    python -m bench.compare_rmsnorm --rows 4096 --cols 4096 --dtype float16 --backend all
"""

from __future__ import annotations

import argparse
import json

import torch

from sifr_ml_systems.bench import cuda_time
from sifr_ml_systems.kernels.rmsnorm import rmsnorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--backend",
        choices=["reference", "triton", "torch.compile", "jax", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    x = torch.randn(args.rows, args.cols, device=device, dtype=dtype)
    weight = torch.ones(args.cols, device=device, dtype=dtype)

    results = {
        "device": str(device),
        "shape": [args.rows, args.cols],
        "dtype": args.dtype,
    }

    want_all = args.backend == "all"

    if want_all or args.backend == "reference":
        results["reference"] = cuda_time(lambda: rmsnorm(x, weight), args.warmup, args.iterations)

    if want_all or args.backend == "triton":
        if device.type == "cuda":
            from sifr_ml_systems.kernels.rmsnorm_triton import rmsnorm_triton

            results["triton_fused"] = cuda_time(lambda: rmsnorm_triton(x, weight), args.warmup, args.iterations)
        else:
            results["triton_fused"] = None

    if want_all or args.backend == "torch.compile":
        from sifr_ml_systems.kernels.rmsnorm_torch import rmsnorm_compiled

        results["torch_compile"] = cuda_time(lambda: rmsnorm_compiled(x, weight), args.warmup, args.iterations)

    if want_all or args.backend == "jax":
        try:
            from jax.dlpack import from_dlpack

            from sifr_ml_systems.kernels.rmsnorm_jax import jax_available, rmsnorm_jax_jit
        except ImportError:
            results["jax_jit"] = None
        else:
            if device.type == "cuda" and jax_available():
                x_jax = from_dlpack(x.contiguous())
                w_jax = from_dlpack(weight.contiguous())
                results["jax_jit"] = cuda_time(lambda: rmsnorm_jax_jit(x_jax, w_jax), args.warmup, args.iterations)
            else:
                results["jax_jit"] = None

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
