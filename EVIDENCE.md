![SIFR identity V2 — the cipher disk at shift zero](.github/assets/readme-header-dark.png)

# Evidence record

## 2026-11-16 — RMSNorm reference vs Triton fused (`bench/compare_rmsnorm.py`)

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (capability 8.9, 7.66 GiB), driver 620.70.06, Linux 7.3.4-zen2-2-zen.

Stack: torch 2.15.0+cu132, triton 3.9, Python 3.12.16, `sifr-ml-systems==0.1.0` (editable).

Protocol: 10 warmup iterations, 100 timed iterations, CUDA events, median reported. Command: `python bench/compare_rmsnorm.py --rows <R> --cols 4096 --dtype <T>`.

| Shape | dtype | reference | Triton fused | speedup |
|---|---|---:|---:|---:|
| 4096 × 4096 | float16 | 3.4365 ms | 0.3164 ms | ≈ 10.9× |
| 8192 × 4096 | float32 | 4.1687 ms | 1.2025 ms | ≈ 3.5× |

Correctness: `pytest -q` → **5 passed** (2026-11-16), including `test_rmsnorm_triton_matches_reference` (CUDA-gated, rtol/atol 2e-3 on float16 inputs).

Scope of evidence: single GPU, single kernel, one shape/dtype pair per run; not a general claim about arbitrary transformer workloads. Re-run the command above to reproduce. The Triton default-FP-fusion warning is recorded; results were produced with Triton defaults.

Method note: wall timings from CUDA events capture kernel time only (not full forward/backward training step time).
