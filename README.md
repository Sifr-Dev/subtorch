![SIFR identity V2 — the cipher disk at shift zero](.github/assets/readme-header-dark.png)

# sifr-ml-systems

Hardware-aware machine-learning systems engineering. The project studies and implements computation below the high-level framework interface: GPU kernels, memory movement, numerical correctness, and measured performance — with reproducible evidence rather than wrapper demos.

## Scope and honesty boundaries

- Correctness is defined by executable tests, not by narrative.
- Benchmarks are reproducible commands with warmup, medians, and explicit hardware/driver reporting.
- GPU claims are only made when the relevant path actually runs on the target GPU. GPU-specific tests skip cleanly on machines without CUDA.
- This is a young project: it currently contains one working, reference-tested fused kernel (RMSNorm). It is not a complete training framework.

## What exists today

| Component | State |
|---|---|
| `sifr_ml_systems.kernels.rmsnorm` | Reference RMSNorm with matching manual-formula test |
| `sifr_ml_systems.kernels.rmsnorm_triton` | Fused Triton kernel, one program per row, single pass |
| `sifr_ml_systems.bench` | CUDA-event benchmark harness (warmup, median) |
| `bench/compare_rmsnorm.py` | Reproducible reference-vs-fused comparison |

## Environment

- Linux, Python 3.12, PyTorch 2.15.
- GPU path: NVIDIA driver + CUDA-enabled PyTorch (benchmarked on an RTX 4060).
- CPU-only machines run the reference tests; the fused kernel test skips cleanly.

## Install

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Run correctness tests

```bash
pytest -q
```

## Run the RMSNorm benchmark

```bash
python bench/compare_rmsnorm.py --rows 4096 --cols 4096 --dtype float16
```

On CPU, only the reference path is timed; the fused kernel requires CUDA and Triton.

## Roadmap (unscripted — recorded as it actually happens)

- [x] RMSNorm: reference + Triton fused kernel, CPU correctness test, CUDA-gated parity test
- [ ] Verified RTX 4060 timings (blocked on this machine's GPU stack installation until recently; see git history)
- [ ] JAX reference path for the same kernel semantics
- [ ] Fused elementwise attention comparison vs PyTorch eager and `scaled_dot_product_attention`
- [ ] cuBLAS `torch.compile` vs hand-written Triton reductions
- [ ] BF16 vs FP32 precision/throughput surface

## License

No license declared yet. The owner will select one before public release.
