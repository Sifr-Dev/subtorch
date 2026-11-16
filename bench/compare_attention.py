"""Reproducible attention benchmark: reference, SDPA, fused Triton.

Usage:
    python -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16
"""

from __future__ import annotations

import argparse
import json

import torch

from sifr_ml_systems.bench import cuda_time
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
    parser.add_argument(
        "--backend",
        choices=["reference", "sdpa", "triton", "all"],
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    q = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device=device, dtype=dtype)
    v = torch.randn(args.batch, args.heads, args.seq, args.head_dim, device=device, dtype=dtype)

    results = {
        "device": str(device),
        "shape": [args.batch, args.heads, args.seq, args.head_dim],
        "dtype": args.dtype,
        "causal": args.causal,
    }

    want_all = args.backend == "all"

    if want_all or args.backend == "reference":
        results["reference"] = cuda_time(
            lambda: attention_reference(q, k, v, causal=args.causal), args.warmup, args.iterations
        )

    if want_all or args.backend == "sdpa":
        results["sdpa"] = cuda_time(
            lambda: attention_sdpa(q, k, v, causal=args.causal), args.warmup, args.iterations
        )

    if want_all or args.backend == "triton":
        if device.type == "cuda":
            from sifr_ml_systems.kernels.attention_triton import attention_triton

            results["triton_fused"] = cuda_time(
                lambda: attention_triton(q, k, v, causal=args.causal), args.warmup, args.iterations
            )
        else:
            results["triton_fused"] = None

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
