"""Tests for paired BEIR component-ablation reporting."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.fitz_bench.beir_ablation import (
    MetricDimension,
    _measurement_failures,
    _shared_dataset_names,
    _subgroup_summaries,
    _variant_command,
    paired_delta,
)
from benchmarks.fitz_bench.beir_holdout import query_manifest_digest
from benchmarks.fitz_bench.retrieval_ablation import get_ablation


def test_paired_delta_is_deterministic_and_reports_positive_interval() -> None:
    before = [0.0, 0.1, 0.2, 0.3, 0.4]
    after = [0.2, 0.3, 0.4, 0.5, 0.6]

    first = paired_delta(before, after, bootstrap_samples=500, seed=17)
    second = paired_delta(before, after, bootstrap_samples=500, seed=17)

    assert first == second
    assert first["mean_delta"] == pytest.approx(0.2)
    assert first["ci95_low"] > 0.0
    assert first["direction"] == "positive"


def test_paired_delta_rejects_unaligned_samples() -> None:
    with pytest.raises(ValueError, match="same positive length"):
        paired_delta([1.0], [1.0, 2.0], bootstrap_samples=100, seed=1)


def test_variant_command_omits_checkpoint_resume_when_disabled() -> None:
    args = Namespace(
        cache_dir=Path("cache"),
        workspace_root=Path("workspace"),
        index_mode="source",
        max_download_gib=2.0,
        max_extracted_gib=4.0,
        resume_queries=False,
        offline=True,
        query_limit=None,
        query_manifest=None,
        governance=None,
        datasets=None,
        cutoffs=[1, 10],
    )

    command = _variant_command(
        args,
        variant="literal",
        output=Path("literal.json"),
        markdown=Path("literal.md"),
    )

    assert "--resume-queries" not in command
    assert "--reuse-index" in command


def test_shared_dataset_names_returns_only_the_ordered_intersection() -> None:
    reports = {
        "literal": {
            "datasets": [
                {"dataset": {"name": "nfcorpus"}},
                {"dataset": {"name": "fiqa"}},
            ]
        },
        "full": {"datasets": [{"dataset": {"name": "fiqa"}}]},
    }

    assert _shared_dataset_names(reports) == ["fiqa"]


def test_measurement_integrity_rejects_wrong_query_manifest() -> None:
    manifest = {"name": "frozen-selection"}
    report = {
        "run": {
            "ablation": get_ablation("literal").as_dict(),
            "query_manifest": {"sha256": "wrong"},
        },
        "complete": True,
        "datasets": [],
    }

    failures = _measurement_failures(
        {"literal": report},
        ["literal"],
        query_manifest=manifest,
    )

    assert failures == ["literal: query manifest metadata mismatch"]
    report["run"]["query_manifest"]["sha256"] = query_manifest_digest(manifest)
    assert (
        _measurement_failures(
            {"literal": report},
            ["literal"],
            query_manifest=manifest,
        )
        == []
    )


def test_subgroup_summaries_pair_records_inside_each_frozen_group() -> None:
    def record(query_id: str, group: str, score: float, latency: float) -> dict:
        metrics = {"NDCG@10": score, "Recall@50": score}
        return {
            "query_id": query_id,
            "holdout": {"group": group},
            "metrics": {
                "recall": dict(metrics),
                "final": dict(metrics),
                "delivered": dict(metrics),
            },
            "latency_seconds": {"fitz_sage": latency},
        }

    results = {
        "literal": {
            "records": [
                record("low-1", "low", 0.0, 1.0),
                record("high-1", "high", 0.5, 1.0),
            ]
        },
        "expansion": {
            "records": [
                record("low-1", "low", 0.2, 2.0),
                record("high-1", "high", 0.5, 2.0),
            ]
        },
        "reranker": {
            "records": [
                record("low-1", "low", 0.0, 2.0),
                record("high-1", "high", 0.5, 2.0),
            ]
        },
        "full": {
            "records": [
                record("low-1", "low", 0.2, 3.0),
                record("high-1", "high", 0.5, 3.0),
            ]
        },
    }
    dimensions = [
        MetricDimension("recall_ndcg", "recall", "NDCG@10"),
        MetricDimension("recall_at_max", "recall", "Recall@50"),
        MetricDimension("final_ndcg", "final", "NDCG@10"),
        MetricDimension("delivered_ndcg", "delivered", "NDCG@10"),
    ]

    summary = _subgroup_summaries(
        "toy",
        results,
        dimensions=dimensions,
        bootstrap_samples=100,
        seed=7,
    )

    assert list(summary) == ["low", "high"]
    assert summary["low"]["effects"]["expansion_without_reranker"]["quality"]["recall_ndcg"][
        "mean_delta"
    ] == pytest.approx(0.2)
    assert (
        summary["high"]["effects"]["expansion_without_reranker"]["quality"]["recall_ndcg"][
            "mean_delta"
        ]
        == 0.0
    )
