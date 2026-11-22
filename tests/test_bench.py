"""CPU-verifiable contracts for benchmark evidence generation."""

import json

import pytest
import torch

import sifr_ml_systems.bench
from sifr_ml_systems.bench import cuda_time, error_metrics, quantile, write_json


def test_cuda_time_cpu_retains_samples_and_dispersion() -> None:
    result = cuda_time(lambda: torch.ones(2), warmup=1, iterations=4, device="cpu")
    assert result["timing"] == "perf_counter_per_iteration"
    assert len(result["samples_ms"]) == 4
    assert result["min_ms"] <= result["median_ms"] <= result["max_ms"]
    assert result["mean_ms"] == pytest.approx(sum(result["samples_ms"]) / 4)
    assert result["p95_ms"] == pytest.approx(quantile(sorted(result["samples_ms"]), 0.95))
    assert "cuda_memory" not in result


def test_cuda_time_rejects_empty_measurement() -> None:
    with pytest.raises(ValueError, match="iterations"):
        cuda_time(lambda: torch.ones(2), warmup=0, iterations=0, device="cpu")


def test_quantile_interpolates_and_median_handles_even_samples() -> None:
    assert quantile([0.0, 10.0], 0.05) == 0.5


def test_error_metrics_reports_actual_bound() -> None:
    expected = torch.tensor([1.0, -2.0])
    out = torch.tensor([1.01, -2.0])
    result = error_metrics(out, expected, rtol=0.02, atol=0.0)
    assert result["passed"] is True
    assert result["max_absolute_error"] == pytest.approx(0.01)
    assert result["max_relative_error_nonzero"] == pytest.approx(0.01)
    assert result["max_bound_excess"] == pytest.approx(-0.01)
    assert result["mismatched_elements"] == 0


def test_error_metrics_rejects_outside_bound() -> None:
    with pytest.raises(AssertionError):
        error_metrics(torch.tensor([1.1]), torch.tensor([1.0]), rtol=0.01, atol=0.0)


def test_error_metrics_empty_tensor() -> None:
    result = error_metrics(torch.empty(0), torch.empty(0), rtol=1e-5, atol=1e-5)
    assert result["num_elements"] == 0
    assert result["mean_absolute_error"] == 0.0


def test_error_metrics_aggregates_across_host_chunks(monkeypatch) -> None:
    # Force one-element host chunks without allocating a huge test tensor.
    monkeypatch.setattr(sifr_ml_systems.bench, "HOST_ERROR_CHUNK_ELEMENTS", 1)
    expected = torch.zeros(2, 2)
    out = torch.tensor([[0.0, 0.0], [0.25, 0.0]])
    result = error_metrics(out, expected, rtol=0.0, atol=1.0)
    assert result["max_absolute_error"] == 0.25
    assert result["mean_absolute_error"] == 0.0625


def test_write_json_is_complete_and_path_explicit(tmp_path, capsys) -> None:
    output = tmp_path / "nested" / "raw.json"
    write_json({"measurement": 1}, output)
    saved = json.loads(output.read_text())
    printed = json.loads(capsys.readouterr().out)
    assert saved["measurement"] == 1
    assert saved["schema_version"] == 1
    assert saved["raw_json_path"] == str(output.resolve())
    assert printed == saved
