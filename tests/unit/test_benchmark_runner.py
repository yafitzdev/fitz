# tests/unit/test_benchmark_runner.py
"""Tests for the retrieval benchmark runner plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchmarks.fitz_bench.models import BenchmarkCase


@dataclass
class _BenchmarkConfig:
    """Small config stand-in with the pydantic API the runner needs."""

    collection: str = "bench"
    governance: str = "pyrrho"

    def model_dump(self) -> dict[str, Any]:
        """Return mutable config values like a pydantic model."""
        return {"collection": self.collection, "governance": self.governance}


def test_create_engine_applies_governance_override(monkeypatch):
    """Benchmark runner should support local Pyrrho model paths."""
    from benchmarks.fitz_bench import runner

    captured: dict[str, Any] = {}

    def fake_load_engine_config(engine: str) -> _BenchmarkConfig:
        captured["loaded_engine"] = engine
        return _BenchmarkConfig()

    def fake_create_engine(engine: str | None, config: Any = None) -> object:
        captured["created_engine"] = engine
        captured["created_config"] = config
        return object()

    monkeypatch.setattr(runner, "load_engine_config", fake_load_engine_config)
    monkeypatch.setattr(runner, "create_engine", fake_create_engine)

    runner._create_engine("fitz_krag", governance=r"pyrrho/C:\models\pyrrho-v2-nano-g1")

    assert captured["loaded_engine"] == "fitz_krag"
    assert captured["created_engine"] == "fitz_krag"
    assert captured["created_config"].governance == r"pyrrho/C:\models\pyrrho-v2-nano-g1"


def test_create_engine_uses_default_config_when_governance_is_not_overridden(monkeypatch):
    """Without an override, the runner should preserve normal runtime loading."""
    from benchmarks.fitz_bench import runner

    captured: dict[str, Any] = {}

    def fail_load_engine_config(engine: str) -> _BenchmarkConfig:
        raise AssertionError("load_engine_config should not run without a governance override")

    def fake_create_engine(engine: str | None) -> object:
        captured["created_engine"] = engine
        return object()

    monkeypatch.setattr(runner, "load_engine_config", fail_load_engine_config)
    monkeypatch.setattr(runner, "create_engine", fake_create_engine)

    runner._create_engine(None, governance=None)

    assert captured["created_engine"] is None


def test_benchmark_workspace_defaults_under_cluster_dir(tmp_path):
    """Benchmark runs should not create many top-level workspace directories."""
    from benchmarks.fitz_bench import runner

    assert (
        runner._benchmark_workspace(tmp_path, None, "bench_123")
        == (tmp_path / ".bench_workspace" / "bench_123").resolve()
    )


def test_benchmark_workspace_resolves_relative_override_under_repo_root(tmp_path):
    """Relative workspace overrides should resolve from the benchmark repo root."""
    from benchmarks.fitz_bench import runner

    assert (
        runner._benchmark_workspace(tmp_path, "custom/workspace", "bench_123")
        == (tmp_path / "custom" / "workspace").resolve()
    )


def test_benchmark_workspace_preserves_absolute_override(tmp_path):
    """Absolute workspace overrides are still supported for manual debugging."""
    from benchmarks.fitz_bench import runner

    override = Path("C:/tmp/fitz-bench-workspace").resolve()
    assert runner._benchmark_workspace(tmp_path, str(override), "bench_123") == override


def test_activate_benchmark_workspace_resets_storage_before_switch(monkeypatch, tmp_path):
    """Sequential production suites must not share the first suite's storage path."""
    from benchmarks.fitz_bench import runner

    calls: list[tuple[str, object | None]] = []
    monkeypatch.setattr(
        runner.SqliteConnectionManager,
        "reset_instance",
        lambda: calls.append(("reset", None)),
    )
    monkeypatch.setattr(
        runner.FitzPaths,
        "set_workspace",
        lambda path: calls.append(("workspace", path)),
    )

    runner._activate_benchmark_workspace(tmp_path)

    assert calls == [("reset", None), ("workspace", tmp_path)]


def test_select_cases_preserves_suite_order():
    """Repeated case filters should not reorder the benchmark suite."""
    from benchmarks.fitz_bench import runner

    cases = [
        BenchmarkCase(case_id="alpha", domain="test", query="A"),
        BenchmarkCase(case_id="beta", domain="test", query="B"),
        BenchmarkCase(case_id="gamma", domain="test", query="C"),
    ]

    selected = runner._select_cases(cases, ["gamma", "alpha"])

    assert [case.case_id for case in selected] == ["alpha", "gamma"]


def test_select_cases_rejects_unknown_id():
    """A typo in a release-gate case filter should fail loudly."""
    from benchmarks.fitz_bench import runner

    cases = [BenchmarkCase(case_id="alpha", domain="test", query="A")]

    with pytest.raises(ValueError, match="missing"):
        runner._select_cases(cases, ["missing"])


def test_record_summary_separates_retrieval_and_governance() -> None:
    from benchmarks.fitz_bench import runner

    records = [
        {
            "validation": {
                "passed": False,
                "metrics": {
                    "retrieval_evaluated": True,
                    "retrieval_passed": True,
                    "delivery_evaluated": True,
                    "delivery_passed": True,
                    "query_shape_evaluated": False,
                    "query_shape_passed": True,
                    "capability_evaluated": True,
                    "capability_passed": True,
                    "mode_match": False,
                    "mrr": 1.0,
                    "required_recall": 1.0,
                    "hit_at_1": True,
                    "hit_at_5": True,
                    "forbidden_count": 0,
                },
            },
            "case": {"required_evidence": [{"file": "policy.md"}]},
        }
    ]

    summary = runner._record_summary(records)

    assert summary["pass_rate"] == 0.0
    assert summary["retrieval_pass_rate"] == 1.0
    assert summary["delivery_pass_rate"] == 1.0
    assert summary["query_shape_pass_rate"] is None
    assert summary["capability_pass_rate"] == 1.0
    assert summary["governance_pass_rate"] == 0.0


def test_gate_uses_selected_metric_and_ingestion_health() -> None:
    from benchmarks.fitz_bench import runner

    summary = {
        "pass_rate": 0.5,
        "capability_pass_rate": 0.9,
        "retrieval_pass_rate": 0.95,
        "delivery_pass_rate": 0.8,
        "query_shape_pass_rate": 0.75,
    }
    healthy = {"summary": {"healthy": True}}
    failed = {"summary": {"healthy": False}}

    assert runner._gate_result(
        summary,
        healthy,
        metric="retrieval",
        minimum=0.85,
        allow_ingestion_failures=False,
    )["passed"]
    assert not runner._gate_result(
        summary,
        failed,
        metric="retrieval",
        minimum=0.85,
        allow_ingestion_failures=False,
    )["passed"]
