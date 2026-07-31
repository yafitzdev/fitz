"""Tests for EnterpriseRAG-Bench paired component reporting."""

from __future__ import annotations

import time
from argparse import Namespace
from pathlib import Path

from benchmarks.fitz_bench.enterprise_rag_ablation import (
    _measurement_failures,
    _variant_command,
    build_ablation_report,
)
from benchmarks.fitz_bench.retrieval_ablation import ablation_names, get_ablation


def test_enterprise_variant_command_reuses_the_frozen_corpus_and_split() -> None:
    args = Namespace(
        cache_dir=Path("cache"),
        workspace_root=Path("workspace"),
        baseline_db=Path("baseline.sqlite3"),
        split_manifest=Path("split.json"),
        selection="development",
        index_mode="source",
        max_download_gib=2.0,
        max_extracted_gib=4.0,
        resume_queries=True,
        offline=True,
        governance=None,
        exclude_relative_paths=["source/quarantined.txt"],
        cutoffs=[10, 50],
    )

    command = _variant_command(
        args,
        variant="full",
        output=Path("full.json"),
        markdown=Path("full.md"),
    )

    assert "benchmarks.fitz_bench.enterprise_rag_benchmark" in command
    assert command[command.index("--selection") + 1] == "development"
    assert "--reuse-index" in command
    assert "--resume-queries" in command
    assert "--offline" in command
    assert command[command.index("--exclude-relative-path") + 1] == ("source/quarantined.txt")


def test_enterprise_ablation_report_pairs_queries_and_has_no_quality_gate(tmp_path) -> None:
    reports = {
        "literal": _variant_report("literal", 0.2, 1.0),
        "expansion": _variant_report("expansion", 0.4, 2.0),
        "reranker": _variant_report("reranker", 0.3, 2.0),
        "full": _variant_report("full", 0.5, 3.0),
    }
    paths = {name: tmp_path / f"{name}.json" for name in reports}

    report = build_ablation_report(
        reports,
        report_paths=paths,
        run_id="run",
        started=time.perf_counter(),
        requested_variants=list(ablation_names()),
        bootstrap_samples=100,
        seed=7,
    )

    assert report["gate"]["passed"] is True
    assert report["method"]["quality_gate"] is False
    effect = report["effects"]["full_vs_literal"]
    assert effect["quality"]["recall_at_max"]["mean_delta"] == 0.3
    assert effect["latency_seconds"]["mean_delta"] == 2.0
    assert report["categories"]["semantic"]["queries"] == 1


def test_enterprise_measurement_integrity_rejects_query_order_drift() -> None:
    reports = {
        "literal": _variant_report("literal", 0.2, 1.0),
        "full": _variant_report("full", 0.5, 3.0),
    }
    reports["full"]["dataset"]["records"][0]["query_id"] = "different"

    failures = _measurement_failures(reports, ["literal", "full"])

    assert failures == ["full: measurement identity differs from literal"]


def _variant_report(variant: str, score: float, latency: float) -> dict:
    metrics = {
        "Recall@10": score,
        "Recall@50": score,
        "NDCG@10": score,
    }
    return {
        "run": {
            "git": {"commit": "abc", "branch": "branch", "dirty": False},
            "selection": "development",
            "split_manifest": {"sha256": "split"},
            "ablation": get_ablation(variant).as_dict(),
        },
        "dataset": {
            "dataset": {"name": "enterprise-rag-bench", "archive_sha256": "archive"},
            "collection": "collection",
            "index_mode": "source",
            "selection": {"cutoffs": [10, 50]},
            "summary": {
                "metrics": {
                    "baseline": {
                        "Recall@10": 0.1,
                        "Recall@50": 0.1,
                        "NDCG@10": 0.1,
                    },
                    "recall": dict(metrics),
                    "final": dict(metrics),
                    "delivered": dict(metrics),
                }
            },
            "fitz_sage": {"latency": {"mean_seconds": latency}},
            "records": [
                {
                    "query_id": "q1",
                    "evaluation": {
                        "category": "semantic",
                        "source_types": ["slack"],
                        "expected_documents": 1,
                    },
                    "metrics": {
                        "recall": dict(metrics),
                        "final": dict(metrics),
                        "delivered": dict(metrics),
                    },
                    "latency_seconds": {"fitz_sage": latency},
                }
            ],
        },
        "gate": {"passed": True},
        "complete": True,
    }
