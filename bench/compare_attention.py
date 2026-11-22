"""Reproducible attention benchmark: reference, SDPA, fused Triton.

Usage:
    python -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16
"""

from __future__ import annotations

import argparse

import torch

from sifr_ml_systems.bench import cuda_time, environment_info, error_metrics, require_cuda, write_json
from sifr_ml_systems.kernels.attention import attention_reference, attention_sdpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--causal", action="store_true", default=False)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", help="Write complete raw JSON to this path (inside the chosen workspace).")
    parser.add_argument(
        "--backend",
        choices=["reference", "sdpa", "triton", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch, args.heads, args.seq, args.head_dim) <= 0:
        raise ValueError("all shape dimensions must be positive")
    if args.seq % 32:
        raise ValueError("seq must be a multiple of 32 for the current Triton kernel")
    dtype = getattr(torch, args.dtype)
    device = require_cuda() if args.device == "cuda" else torch.device("cpu")

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device=device, dtype=dtype)

    results = {
        "benchmark": "attention",
        "environment": environment_info(device),
        "seed": 0,
        "shape": [args.batch, args.heads, args.seq, args.head_dim],
        "dtype": args.dtype,
        "causal": args.causal,
    }

    want_all = args.backend == "all"
    expected = attention_reference(q, k, v, causal=args.causal)
    rtol = atol = 1e-5 if dtype == torch.float32 else 2e-2

    def measure(name: str, fn) -> None:
        out = fn()
        result = cuda_time(fn, args.warmup, args.iterations, device=device)
        result["parity_vs_reference"] = error_metrics(out, expected, rtol, atol)
        results[name] = result

    if want_all or args.backend == "reference":
        measure("reference", lambda: attention_reference(q, k, v, causal=args.causal))

    if want_all or args.backend == "sdpa":
        measure("sdpa", lambda: attention_sdpa(q, k, v, causal=args.causal))

    if want_all or args.backend == "triton":
        if device.type == "cuda":
            from sifr_ml_systems.kernels.attention_triton import attention_triton

            measure("triton_fused", lambda: attention_triton(q, k, v, causal=args.causal))
        else:
            results["triton_fused"] = {"status": "not_run", "reason": "CUDA required"}

    write_json(results, args.output)


if __name__ == "__main__":
    main()
