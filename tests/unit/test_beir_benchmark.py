"""Tests for BEIR-to-RetrievalRun identity mapping."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.fitz_bench.beir_benchmark import (
    _append_checkpoint,
    _initialize_checkpoint,
    _load_checkpoint,
    _markdown,
    _run_rankings,
    _stage_retrieval_summary,
    _source_id_mapping,
)


def test_source_id_mapping_joins_manifest_to_external_document_ids(tmp_path) -> None:
    mapping = tmp_path / "mapping.jsonl"
    mapping.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "relative_path": "ab/hash.txt",
                "content_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    entry = SimpleNamespace(file_id="source-1")
    manifest = SimpleNamespace(entries=lambda: {"ab/hash.txt": entry})

    source_ids, summary = _source_id_mapping(manifest, mapping)

    assert source_ids == {"source-1": "doc-1"}
    assert summary["complete"] is True


def test_run_rankings_deduplicates_chunks_and_counts_unmapped_candidates() -> None:
    run = SimpleNamespace(
        candidate_stages=(
            SimpleNamespace(
                name="recall",
                candidates=(
                    SimpleNamespace(source_id="source-1"),
                    SimpleNamespace(source_id="source-1"),
                    SimpleNamespace(source_id="synthetic"),
                ),
            ),
        ),
        ranked_evidence=(SimpleNamespace(source_id="source-1"),),
        pyrrho_evidence=(SimpleNamespace(source_id="source-1"),),
    )

    rankings, unmapped = _run_rankings(run, {"source-1": "doc-1"})

    assert rankings["recall"] == ["doc-1"]
    assert rankings["compiled"] == ["doc-1"]
    assert rankings["delivered"] == ["doc-1"]
    assert rankings["reranked"] == []
    assert unmapped["recall"] == 1


def test_query_checkpoint_round_trips_completed_records(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    signature = {"dataset": "toy", "source": "abc"}
    record = {"query_id": "q1", "metrics": {"Recall@5": 1.0}}
    _initialize_checkpoint(path, signature)
    _append_checkpoint(path, record)

    assert _load_checkpoint(path, signature) == {"q1": record}


def test_query_checkpoint_ignores_only_truncated_final_record(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    signature = {"dataset": "toy"}
    _initialize_checkpoint(path, signature)
    _append_checkpoint(path, {"query_id": "q1"})
    with path.open("a", encoding="utf-8") as output:
        output.write('{"type":"query"')

    assert _load_checkpoint(path, signature) == {"q1": {"query_id": "q1"}}


def test_query_checkpoint_rejects_stale_signature(tmp_path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    _initialize_checkpoint(path, {"dataset": "old"})

    with pytest.raises(ValueError, match="does not match"):
        _load_checkpoint(path, {"dataset": "new"})


def test_stage_retrieval_summary_reports_depth_and_relevant_hit_rate() -> None:
    records = [
        {
            "rankings": {"recall": ["a", "b"]},
            "judgments": {"b": 1},
        },
        {
            "rankings": {"recall": ["c"]},
            "judgments": {"d": 2},
        },
    ]

    assert _stage_retrieval_summary(records, "recall") == {
        "mean_unique_documents": 1.5,
        "relevant_hit_rate": 0.5,
    }


def test_markdown_uses_an_actually_requested_ranking_cutoff() -> None:
    report = {
        "run": {"run_id": "run", "index_mode": "source"},
        "gate": {"passed": True},
        "datasets": [
            {
                "dataset": {"name": "toy", "corpus_documents": 1},
                "selection": {"queries": 1, "cutoffs": [3, 50]},
                "summary": {
                    "metrics": {
                        "baseline": {
                            "NDCG@3": 0.5,
                            "Recall@50": 1.0,
                            "MRR@3": 0.5,
                        }
                    }
                },
            }
        ],
    }

    rendered = _markdown(report)

    assert "NDCG@3" in rendered
    assert "MRR@3" in rendered
    assert "NDCG@10" not in rendered
