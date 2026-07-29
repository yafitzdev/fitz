"""Tests for the reusable BEIR stage-timing profiler."""

from __future__ import annotations

import pytest

from benchmarks.fitz_bench.beir_timing import (
    _group_timings,
    _select_query_ids,
    _summarize_records,
)


def test_group_timings_ignores_overlapping_retrieval_totals() -> None:
    grouped, overlap = _group_timings(
        {
            "Qwen query keywords": 2.0,
            "Recall": 0.5,
            "Rerank": 5.0,
            "Read": 0.25,
            "Retrieval": 6.0,
            "Evidence closure 1 Recall": 0.2,
            "Evidence closure 1 Rerank": 1.5,
            "Evidence closure 1 Read": 0.1,
            "Evidence closure 1": 2.0,
            "Pyrrho": 1.0,
        },
        total_seconds=11.0,
    )

    assert grouped["semantic_expansion"] == 2.0
    assert grouped["recall"] == 0.7
    assert grouped["rerank"] == 6.5
    assert grouped["read"] == 0.35
    assert grouped["pyrrho_decision"] == 1.0
    assert grouped["unattributed"] == pytest.approx(0.45)
    assert overlap == 0.0


def test_group_timings_reports_overlap_instead_of_negative_residual() -> None:
    grouped, overlap = _group_timings(
        {"Qwen query keywords": 2.0, "Pyrrho": 1.0},
        total_seconds=2.5,
    )

    assert grouped["unattributed"] == 0.0
    assert overlap == 0.5


def test_select_query_ids_is_stable_and_preserves_source_order() -> None:
    query_ids = [f"q{index}" for index in range(20)]

    selected = _select_query_ids(query_ids, sample_size=5, seed=42)

    assert selected == _select_query_ids(query_ids, sample_size=5, seed=42)
    assert selected == sorted(selected, key=query_ids.index)
    assert len(selected) == 5


def test_summarize_records_reports_exclusive_stage_share() -> None:
    records = [
        {
            "total_seconds": 10.0,
            "grouped_seconds": {
                "semantic_expansion": 2.0,
                "rerank": 5.0,
                "unattributed": 3.0,
            },
            "stage_seconds": {
                "Qwen query keywords": 2.0,
                "Rerank": 5.0,
            },
        },
        {
            "total_seconds": 20.0,
            "grouped_seconds": {
                "semantic_expansion": 4.0,
                "rerank": 10.0,
                "unattributed": 6.0,
            },
            "stage_seconds": {
                "Qwen query keywords": 4.0,
                "Rerank": 10.0,
            },
        },
    ]

    summary = _summarize_records(records)

    assert summary["total_latency"]["mean_seconds"] == 15.0
    assert summary["stage_groups"]["semantic_expansion"]["mean_seconds"] == 3.0
    assert summary["stage_groups"]["semantic_expansion"]["share_of_total"] == 0.2
    assert summary["stage_groups"]["rerank"]["share_of_total"] == 0.5
    assert summary["raw_stages"]["Rerank"]["observations"] == 2
