# Darkmatter

Hardware-aware machine-learning kernels with reproducible, cross-framework evidence.

<p>
  <a href="https://github.com/Sifr-Dev/Darkmatter/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Sifr-Dev/Darkmatter/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/PYTHON-3.12-3776ab?logo=python&logoColor=white">
  <img alt="PyTorch 2.15" src="https://img.shields.io/badge/PYTORCH-2.15-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Tests: 54" src="https://img.shields.io/badge/TESTS-54-39c5cf">
  <img alt="License: MIT" src="https://img.shields.io/badge/LICENSE-MIT-2ea043">
</p>

> Distribution/package name remains `sifr-ml-systems` and the import package
> remains `sifr_ml_systems`; the product name is Darkmatter.

## What it is

Darkmatter studies and implements computation below the high-level framework
interface: GPU kernels, memory movement, numerical correctness, and measured
performance. It currently contains two reference-tested fused kernels — RMSNorm
and FlashAttention-2-style attention — compared against PyTorch, `torch.compile`,
JAX and SDPA. It is a kernel and evidence project, not a training framework.

## Current capabilities

- **RMSNorm reference with a matching manual-formula test** —
  `src/sifr_ml_systems/kernels/rmsnorm.py`, `tests/test_rmsnorm.py` (5 tests).
- **Fused Triton RMSNorm** (one program per row, single pass) —
  `src/sifr_ml_systems/kernels/rmsnorm_triton.py`; fp32/fp16/bf16 parity in
  `tests/test_rmsnorm_parity.py` (13 tests).
- **`torch.compile` and JAX/XLA baselines for the same semantics** —
  `rmsnorm_torch.py`, `rmsnorm_jax.py`; parity is part of the same suite. JAX is
  an optional dependency group and skips cleanly when absent.
- **Fused attention forward with online softmax and causal masking** —
  `src/sifr_ml_systems/kernels/attention_triton.py`; Triton vs naive reference vs
  SDPA parity across dtypes, causal/non-causal and shapes in
  `tests/test_attention_parity.py` (18 tests).
- **CPU correctness path** — `tests/test_attention_cpu.py` (10 tests) and the
  ungated reference tests; `pytest -m "not gpu"` runs 22 tests without CUDA.
- **CUDA-event benchmark harness with full JSON output** — `bench/compare_rmsnorm.py`,
  `bench/compare_attention.py`; records version/driver/GPU metadata, every
  sample, min/median/mean/max, p05/p95, allocator peaks and error bounds.
- **CI that separates CPU from GPU** — `.github/workflows/ci.yml`: a hosted
  CPU-only job, an opt-in self-hosted RTX 4060 job, and
  `.github/scripts/check-cpu-coverage.py` which fails if the CPU job stops
  covering a CPU-capable kernel module.
- **Recorded evidence** — `EVIDENCE.md` plus raw JSON and the reproduction log in
  `evidence/2026-11-21-rtx4060/`.

## Quick start

```sh
uv python install 3.12
uv sync --python 3.12 --frozen --extra bench --group dev --no-group jax
.venv/bin/python -m pytest -q                 # CPU + GPU when CUDA is available
.venv/bin/python -m pytest -q -m "not gpu"    # CPU only
```

A deliberately CPU-only environment (also proves CUDA tests are not silently
passing on a CUDA wheel):

```sh
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python torch==2.15.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python -e . --group dev
.venv-cpu/bin/python -c 'import torch; assert not torch.cuda.is_available()'
.venv-cpu/bin/python -m pytest -q -m "not gpu"
```

## Verification

Run on this host (NVIDIA RTX 4060 Laptop, driver 620.70.06, Python 3.12.16,
torch 2.15.0+cu132, Triton 3.9.0):

```sh
.venv/bin/python -m pytest -q
.venv-cpu/bin/python -m pytest -q -m "not gpu"
.venv-cpu/bin/python .github/scripts/check-cpu-coverage.py
```

Observed: **54 passed, 0 failed** in the CUDA environment (22 CPU + 32 GPU
parity tests); **22 passed, 32 deselected** in the CPU-only environment; the
coverage check reports `22 CPU-selected tests cover 3 kernel modules` and exits
0. A skipped GPU test is not evidence of GPU correctness.

## Benchmarks

Measured medians, RTX 4060 Laptop (8.0 GiB, capability 8.9), driver 620.70.06,
torch 2.15.0+cu132, Triton 3.9.0, CUDA 13.2. Warm-up 10 iterations, 100 timed
iterations for RMSNorm, CUDA events, median of every retained sample. Raw JSON is
in `evidence/2026-11-21-rtx4060/raw/`; the earlier tables and reproduction
commands are in `EVIDENCE.md`.

RMSNorm, 4096×4096, float16:

| Backend | Median |
|---|---:|
| PyTorch reference (unfused) | 3.706 ms |
| Triton fused | 0.330 ms |
| `torch.compile` | 0.337 ms |
| JAX/XLA (`jit`) | 0.534 ms |

Attention forward, batch 4 × 16 heads × 1024 seq × 64 head-dim, float16:

| Configuration | Reference | SDPA | Triton fused |
|---|---:|---:|---:|
| non-causal | 9.25 ms | 0.614 ms | 0.617 ms |
| causal | 12.92 ms | **0.372 ms** | 0.632 ms |

Limitations: one GPU, one set of shapes, kernel-only time from CUDA events (not
full training-step wall time), and allocator peaks are process-relative. Triton
matches SDPA within ~1% on the non-causal fp16 case but trails it on causal and
on head-dim 128; the kernel is correct, not vendor-tuned. JAX numbers differ
between recorded runs; no universal backend ranking is claimed.

## Architecture

```
src/sifr_ml_systems/kernels/rmsnorm.py          reference RMSNorm
src/sifr_ml_systems/kernels/rmsnorm_triton.py   fused Triton kernel
src/sifr_ml_systems/kernels/rmsnorm_torch.py    torch.compile baseline
src/sifr_ml_systems/kernels/rmsnorm_jax.py      JAX reference / jit
src/sifr_ml_systems/kernels/attention.py        naive attention reference
src/sifr_ml_systems/kernels/attention_triton.py online-softmax fused attention
src/sifr_ml_systems/bench.py                   benchmark measurement helpers
bench/                                          CLI comparators, JSON output
tests/                                          CPU and CUDA-gated parity suites
evidence/                                       raw benchmark and correctness JSON
```

## Limitations

- Two kernels only; no training loop, optimizer, dataloader or model runtime.
- GPU correctness is proven on the tested RTX 4060 class device only; other
  architectures are unverified.
- Benchmarks are kernel-level and shape-specific; do not extrapolate them to
  end-to-end transformer throughput.
- The CUDA CI job requires a self-hosted runner and is opt-in, not a hosted-GPU claim.
- JAX is optional and absent from the default environment; JAX tests and the JAX
  benchmark skip when it is not installed.

## Roadmap (NOT IMPLEMENTED)

- cuBLAS/`torch.compile` vs hand-written Triton reductions on more shapes.
- A systematic BF16 vs FP32 precision/throughput surface.
- Additional kernels (layernorm variants, softmax, GEMM epilogues).
- Multi-GPU and non-NVIDIA architecture coverage.

## License

MIT — see `LICENSE`.
