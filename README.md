![SIFR identity V2 — the cipher disk at shift zero](.github/assets/readme-header-dark.png)

# sifr-ml-systems

Hardware-aware machine-learning systems engineering. The project studies and implements computation below the high-level framework interface: GPU kernels, memory movement, numerical correctness, and measured performance — with reproducible evidence rather than wrapper demos.

## Scope and honesty boundaries

- Correctness is defined by executable tests, not by narrative.
- Benchmarks are reproducible commands with warmup, medians, and explicit hardware/driver reporting.
- GPU claims are only made when the relevant path actually runs on the target GPU. GPU-specific tests skip cleanly on machines without CUDA.
- This is a young project: it now contains two working, reference-tested fused kernels (RMSNorm, fused attention) compared across torch, Triton, torch.compile, JAX and SDPA. It is not a complete training framework. On this GPU and these shapes, the JAX/XLA path was measured fastest for RMSNorm and PyTorch's fused SDPA leads on causal attention; see `EVIDENCE.md`.

## What exists today

| Component | State |
|---|---|
| `sifr_ml_systems.kernels.rmsnorm` | Reference RMSNorm with matching manual-formula test |
| `sifr_ml_systems.kernels.rmsnorm_triton` | Fused Triton kernel, one program per row, single pass |
| `sifr_ml_systems.kernels.rmsnorm_jax` | JAX reference (`rmsnorm_jax`) and `rmsnorm_jax_jit` (XLA-compiled) |
| `sifr_ml_systems.kernels.rmsnorm_torch` | `torch.compile`'d reference baseline |
| `sifr_ml_systems.bench` | CUDA-event benchmark harness (warmup, median) |
| `bench/compare_rmsnorm.py` | Reproducible four-backend comparison (`--backend all`) |
| `tests/test_rmsnorm_parity.py` | fp32/fp16/bf16 parity: Triton vs reference vs torch.compile vs JAX |
| `sifr_ml_systems.kernels.attention_triton` | FlashAttention-2-style fused attention forward (online softmax, causal support) |
| `bench/compare_attention.py` | Reproducible attention comparison (`--backend all`) |
| `tests/test_attention_parity.py` | fp32/fp16/bf16 parity × causal × shapes: Triton vs reference vs SDPA |

## Environment

- Linux, Python 3.12, PyTorch 2.15.
- GPU path: NVIDIA driver + CUDA-enabled PyTorch (benchmarked on an RTX 4060).
- CPU-only machines run the reference tests; the fused kernel test skips cleanly.

## Reproducible environment

The lock file records exact package versions and hashes. These commands use `uv`'s managed Python 3.12 and keep both environments inside this repository. No Nix or system Python changes are needed.

```bash
uv python install 3.12
uv sync --python 3.12 --frozen --extra bench --group dev --no-group jax
# Optional, separately resolved dependency group (CUDA 13 build):
uv sync --python 3.12 --frozen --extra bench --group dev --group jax
```

The base environment contains `torch==2.15.0`, `triton==3.9.0`, and `numpy==2.5.5`. The separately selectable JAX group resolves compatibly with these versions (`jax==0.12.1`). `uv sync --group jax` replaces the environment's selected dependency set, so rerun the base `--no-group jax` command to remove it. JAX is never needed for the reference/Triton/SDPA tests.

For a deliberately CPU-only test environment (also proves CUDA tests are not silently skipped by an unexpected CUDA wheel):

```bash
uv venv --python 3.12 .venv-cpu
uv pip install --python .venv-cpu/bin/python torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-cpu/bin/python -e . --group dev
.venv-cpu/bin/python -c 'import torch; assert not torch.cuda.is_available(); assert torch.version.cuda is None'
.venv-cpu/bin/python -m pytest -q -m "not gpu"
```

`pytest -q` in a CUDA environment runs CPU and GPU parity tests. `pytest -q -m gpu` explicitly selects GPU tests. A skipped GPU test is not evidence of GPU correctness.

## Benchmarks: complete raw evidence

The benchmark commands below require an actual CUDA device; they fail instead of replacing GPU measurements with CPU results. Each emits complete JSON, and `--output` persists it. Records include Python/PyTorch/Triton versions, GPU name/capability/memory/driver, per-iteration CUDA-event timings, min/median/mean/max, sample standard deviation, p05/p95, PyTorch CUDA allocator peaks, and actual numerical error bounds.

```bash
.venv/bin/python -m bench.compare_rmsnorm --rows 4096 --cols 4096 --dtype float16 \
  --backend all --warmup 10 --iterations 100 \
  --output evidence/rtx4060/rmsnorm-fp16.json
.venv/bin/python -m bench.compare_attention --batch 4 --heads 16 --seq 1024 --head-dim 64 \
  --dtype float16 --backend all --warmup 10 --iterations 50 \
  --output evidence/rtx4060/attention-fp16.json
# Add --causal for the causal-attention configuration.
```

The parity check runs before timing, so initial JIT compilation is outside both warm-up and timed samples. JAX calls are explicitly awaited for completion. CUDA events measure callable execution on the current CUDA stream, not full training-step wall time. Peak allocator statistics are process-relative, reset after warm-up, and are not exclusive whole-GPU memory usage. `samples_ms` retains every sample rather than reporting only a median. The original RMSNorm benchmark used `weight = ones`; this is retained for comparability, while the CUDA parity tests exercise varying weights.

CI's `CPU-only correctness` job uses an explicit CPU-only wheel on GitHub-hosted Ubuntu. The `CUDA correctness` job is manual-dispatch-only and requires a self-hosted runner labeled `gpu` and `rtx4060`; it is not a hosted-GPU claim. See `EVIDENCE.md` for measured local results and raw JSON paths.

## Roadmap (unscripted — recorded as it actually happens)

- [x] RMSNorm: reference + Triton fused kernel, CPU correctness test, CUDA-gated parity test
- [x] Verified RTX 4060 timings (see `EVIDENCE.md`, 2026-11-16)
- [x] JAX 0.12.0.1 + CUDA 13 plugin recognized the GPU (`CudaDevice(id=0)`)
- [x] JAX reference path for the same kernel semantics (`rmsnorm_jax`, `rmsnorm_jax_jit`)
- [x] Cross-implementation parity tests (Triton / torch.compile / JAX vs reference) — 2026-11-16, `pytest -q` 18 passed
- [x] Four-backend benchmark recorded with reproduction commands (`EVIDENCE.md`, 2026-11-16)
- [x] Fused attention (FlashAttention-2-style online softmax, causal support) vs naive reference and SDPA — `EVIDENCE.md`, 2026-11-16
- [ ] cuBLAS `torch.compile` vs hand-written Triton reductions
- [ ] BF16 vs FP32 precision/throughput surface

## License

No license declared yet. The owner will select one before public release.
