"""Tests for the production benchmark matrix aggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.fitz_bench import runner
from benchmarks.fitz_bench.production_runner import (
    _aggregate,
    _governance_identity,
    _select_suites,
)


def _report(*, required: bool, passed: bool, retrieval_passed: int) -> dict:
    return {
        "matrix": {
            "required": required,
            "compare_to": None,
        },
        "gate": {"passed": passed},
        "ingestion": {"summary": {"healthy": True}},
        "summary": {
            "total": 2,
            "passed": 1,
            "pass_rate": 0.5,
            "retrieval_evaluated": 2,
            "retrieval_passed": retrieval_passed,
            "retrieval_pass_rate": retrieval_passed / 2,
            "delivery_evaluated": 2,
            "delivery_passed": retrieval_passed,
            "delivery_pass_rate": retrieval_passed / 2,
            "query_shape_evaluated": 0,
            "query_shape_passed": 0,
            "query_shape_pass_rate": None,
            "capability_evaluated": 2,
            "capability_passed": retrieval_passed,
            "capability_pass_rate": retrieval_passed / 2,
        },
        "records": [],
    }


def test_aggregate_weights_required_suites_only(tmp_path) -> None:
    aggregate = _aggregate(
        suite_path=tmp_path / "production.yaml",
        reports={
            "required": _report(required=True, passed=True, retrieval_passed=2),
            "measurement": _report(required=False, passed=False, retrieval_passed=0),
        },
        duration=12.0,
    )

    assert aggregate["summary"]["required_cases"] == 2
    assert aggregate["summary"]["retrieval_pass_rate"] == 1.0
    assert aggregate["summary"]["production_gate_passed"] is True


def test_select_suites_rejects_unknown_ids() -> None:
    with pytest.raises(ValueError, match="missing"):
        _select_suites([{"id": "core"}], ["missing"])


def test_query_shape_suite_has_balanced_positive_and_negative_controls() -> None:
    root = Path(__file__).resolve().parents[2]
    cases = runner._load_cases(root / "benchmarks" / "cases" / "query_shapes.yaml")

    assert len(cases) == 60
    assert all(not case.required_evidence for case in cases)
    assert all(not case.forbidden_evidence for case in cases)
    assert all(len(case.expected_signals) == 3 for case in cases)
    for tag in ("temporal", "comparison", "aggregation", "narrow"):
        assert sum(tag in case.tags for case in cases) == 15


def test_governance_identity_records_local_graph_and_release_status(tmp_path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"fp32-graph")
    (tmp_path / "onnx_parity_report.json").write_text(
        '{"passed": true, "comparisons": {"native_vs_fp32": '
        '{"passed": true, "decision_differences": 0}}}',
        encoding="utf-8",
    )

    identity = _governance_identity(f"pyrrho/{tmp_path}")

    assert identity["local_package"] is True
    assert identity["manifest"]["present"] is False
    assert identity["selected_graph"]["present"] is True
    assert identity["selected_graph"]["sha256"]
    assert identity["parity_report"]["passed"] is True
