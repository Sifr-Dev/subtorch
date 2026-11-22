"""Reproducible RMSNorm benchmark: reference, Triton, torch.compile, JAX.

Usage:
    python -m bench.compare_rmsnorm --rows 4096 --cols 4096 --dtype float16
    python -m bench.compare_rmsnorm --rows 4096 --cols 4096 --dtype float16 --backend all
"""

from __future__ import annotations

import argparse

import torch

from sifr_ml_systems.bench import cuda_time, environment_info, error_metrics, require_cuda, write_json
from sifr_ml_systems.kernels.rmsnorm import rmsnorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--cols", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", help="Write complete raw JSON to this path (inside the chosen workspace).")
    parser.add_argument(
        "--backend",
        choices=["reference", "triton", "torch.compile", "jax", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive")
    dtype = getattr(torch, args.dtype)
    device = require_cuda() if args.device == "cuda" else torch.device("cpu")

    torch.manual_seed(0)
    x = torch.randn(args.rows, args.cols, device=device, dtype=dtype)
    weight = torch.ones(args.cols, device=device, dtype=dtype)

    results = {
        "benchmark": "rmsnorm",
        "environment": environment_info(device),
        "seed": 0,
        "eps": 1e-6,
        "shape": [args.rows, args.cols],
        "dtype": args.dtype,
    }

    want_all = args.backend == "all"
    expected = rmsnorm(x, weight)
    rtol = atol = 1e-5 if dtype == torch.float32 else 2e-2

    def measure(name: str, fn, convert=torch.Tensor) -> None:
        out = convert(fn())
        result = cuda_time(fn, args.warmup, args.iterations, device=device)
        result["parity_vs_reference"] = error_metrics(out, expected, rtol, atol)
        results[name] = result

    if want_all or args.backend == "reference":
        measure("reference", lambda: rmsnorm(x, weight))

    if want_all or args.backend == "triton":
        if device.type == "cuda":
            from sifr_ml_systems.kernels.rmsnorm_triton import rmsnorm_triton

            measure("triton_fused", lambda: rmsnorm_triton(x, weight))
        else:
            results["triton_fused"] = {"status": "not_run", "reason": "CUDA required"}

    if want_all or args.backend == "torch.compile":
        from sifr_ml_systems.kernels.rmsnorm_torch import rmsnorm_compiled

        measure("torch_compile", lambda: rmsnorm_compiled(x, weight))

    if want_all or args.backend == "jax":
        try:
            from jax.dlpack import from_dlpack

            from sifr_ml_systems.kernels.rmsnorm_jax import jax_available, rmsnorm_jax_jit
        except ImportError:
            results["jax_jit"] = {"status": "not_run", "reason": "optional JAX dependency group not installed"}
        else:
            if device.type == "cuda" and jax_available():
                x_jax = from_dlpack(x.contiguous())
                w_jax = from_dlpack(weight.contiguous())
                jax_out = torch.from_dlpack(rmsnorm_jax_jit(x_jax, w_jax))
                parity = error_metrics(jax_out, expected, rtol, atol)
                del expected, jax_out, parity
                torch.cuda.empty_cache()

                def run_jax():
                    return rmsnorm_jax_jit(x_jax, w_jax).block_until_ready()

                result = cuda_time(run_jax, args.warmup, args.iterations, device=device)
                result["parity_vs_reference"] = error_metrics(
                    torch.from_dlpack(run_jax()), rmsnorm(x, weight), rtol, atol
                )
                results["jax_jit"] = result
            else:
                results["jax_jit"] = {"status": "not_run", "reason": "CUDA-capable JAX required"}

    write_json(results, args.output)


if __name__ == "__main__":
    main()
