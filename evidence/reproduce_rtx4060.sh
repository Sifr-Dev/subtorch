#!/usr/bin/env bash
# Run from the repository root. GPU commands fail if CUDA is unavailable.
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"
OUT="${OUT:-evidence/2026-11-21-rtx4060/raw}"
mkdir -p "$OUT"

"$PYTHON" -c 'import json, platform, torch; assert torch.cuda.is_available(), "CUDA unavailable: no GPU tests or benchmarks will be reported as run"; print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}, indent=2))'
"$PYTHON" -m pytest -q
# With the optional CUDA-JAX group installed, measure all four backends. Without
# it, run reference and Triton only and report JAX as not installed.
if "$PYTHON" -c 'import jax' 2>/dev/null; then
  RMS_BACKENDS=(reference triton torch.compile jax)
else
  printf 'JAX not installed: skipping optional JAX benchmark; run uv sync --group jax to enable it.\n'
  RMS_BACKENDS=(reference triton)
fi
for backend in "${RMS_BACKENDS[@]}"; do
  "$PYTHON" -m bench.compare_rmsnorm --rows 4096 --cols 4096 --dtype float16 --backend "$backend" --warmup 10 --iterations 100 --output "$OUT/rmsnorm-$backend-4096x4096-fp16.json"
  "$PYTHON" -m bench.compare_rmsnorm --rows 8192 --cols 4096 --dtype float32 --backend "$backend" --warmup 10 --iterations 100 --output "$OUT/rmsnorm-$backend-8192x4096-fp32.json"
done
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16 --backend reference --warmup 10 --iterations 50 --output "$OUT/attention-reference-4x16x1024x64-fp16.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16 --backend sdpa --warmup 10 --iterations 50 --output "$OUT/attention-sdpa-4x16x1024x64-fp16.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16 --backend triton --warmup 10 --iterations 50 --output "$OUT/attention-triton-4x16x1024x64-fp16.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16 --causal --backend reference --warmup 10 --iterations 50 --output "$OUT/attention-reference-4x16x1024x64-fp16-causal.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16 --causal --backend sdpa --warmup 10 --iterations 50 --output "$OUT/attention-sdpa-4x16x1024x64-fp16-causal.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype float16 --causal --backend triton --warmup 10 --iterations 50 --output "$OUT/attention-triton-4x16x1024x64-fp16-causal.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype bfloat16 --backend reference --warmup 10 --iterations 50 --output "$OUT/attention-reference-4x16x1024x64-bf16.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype bfloat16 --backend sdpa --warmup 10 --iterations 50 --output "$OUT/attention-sdpa-4x16x1024x64-bf16.json"
"$PYTHON" -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 --dtype bfloat16 --backend triton --warmup 10 --iterations 50 --output "$OUT/attention-triton-4x16x1024x64-bf16.json"
"$PYTHON" -m bench.compare_attention --batch 2 --heads 8 --seq 2048 --head-dim 128 --dtype float16 --backend reference --warmup 10 --iterations 50 --output "$OUT/attention-reference-2x8x2048x128-fp16.json"
"$PYTHON" -m bench.compare_attention --batch 2 --heads 8 --seq 2048 --head-dim 128 --dtype float16 --backend sdpa --warmup 10 --iterations 50 --output "$OUT/attention-sdpa-2x8x2048x128-fp16.json"
"$PYTHON" -m bench.compare_attention --batch 2 --heads 8 --seq 2048 --head-dim 128 --dtype float16 --backend triton --warmup 10 --iterations 50 --output "$OUT/attention-triton-2x8x2048x128-fp16.json"
