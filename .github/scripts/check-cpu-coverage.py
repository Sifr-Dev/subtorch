#!/usr/bin/env python3
"""Fail if a CPU-only run silently stops covering a kernel module.

32 of this project's 54 tests are gated on `torch.cuda.is_available()`, so the
CPU-only job is the only thing that runs on a stock runner. If a module loses
its last ungated test, the job still passes while covering nothing. This check
requires that every module under src/sifr_ml_systems/kernels/ that a CPU test
can import is actually imported by at least one selected CPU test.

Run it with the CPU interpreter so marker selection matches CI:

    .venv-cpu/bin/python .github/scripts/check-cpu-coverage.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Kernel modules that are inherently CUDA-only: a CPU job cannot cover them.
CUDA_ONLY = {
    "attention_triton.py",
    "rmsnorm_triton.py",
    "rmsnorm_jax.py",
    "rmsnorm_torch.py",
}


def cpu_selected_tests(python: str) -> list[str]:
    result = subprocess.run(
        [python, "-m", "pytest", "--collect-only", "-q", "-m", "not gpu"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"pytest collection failed:\n{result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if "::" in line]


def modules_imported_by(test_ids: list[str]) -> set[str]:
    """Map selected test files to the kernel modules they import."""
    repo = Path(__file__).resolve().parents[2]
    kernels = repo / "src" / "sifr_ml_systems" / "kernels"
    imported: set[str] = set()

    files = {test_id.split("::")[0] for test_id in test_ids}
    for name in files:
        path = repo / name
        if not path.is_file():
            continue
        text = path.read_text()
        for match in re.finditer(r"kernels\.(\w+)", text):
            module = match.group(1)
            for candidate in (f"{module}.py", f"{module}"):
                if (kernels / candidate).is_file():
                    imported.add(candidate)
    return imported


def main() -> int:
    python = sys.executable
    test_ids = cpu_selected_tests(python)
    if not test_ids:
        sys.exit("error: the CPU-only selection matched no tests")

    repo = Path(__file__).resolve().parents[2]
    kernels = repo / "src" / "sifr_ml_systems" / "kernels"
    if not kernels.is_dir():
        sys.exit(f"error: {kernels} not found")

    covered = modules_imported_by(test_ids)
    gaps = []
    for path in sorted(kernels.glob("*.py")):
        if path.name == "__init__.py" or path.name in CUDA_ONLY:
            continue
        if path.name not in covered:
            gaps.append(path.name)

    if gaps:
        for gap in gaps:
            print(
                f"error: {gap} has no CPU-selected test; a CPU-only run would "
                "pass without exercising it",
                file=sys.stderr,
            )
        return 1

    print(f"{len(test_ids)} CPU-selected tests cover {len(covered)} kernel modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
