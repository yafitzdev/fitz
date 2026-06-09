# tests/unit/test_benchmark_runner.py
"""Tests for the retrieval benchmark runner plumbing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _BenchmarkConfig:
    """Small config stand-in with the pydantic API the runner needs."""

    collection: str = "bench"
    governance: str = "pyrrho"

    def model_dump(self) -> dict[str, Any]:
        """Return mutable config values like a pydantic model."""
        return {"collection": self.collection, "governance": self.governance}


def test_create_engine_applies_governance_override(monkeypatch):
    """Benchmark runner should support local Pyrrho package paths."""
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

    runner._create_engine("fitz_krag", governance=r"pyrrho/C:\models\pyrrho-nano-g4-alpha")

    assert captured["loaded_engine"] == "fitz_krag"
    assert captured["created_engine"] == "fitz_krag"
    assert captured["created_config"].governance == r"pyrrho/C:\models\pyrrho-nano-g4-alpha"


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
