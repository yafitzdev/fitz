"""Tests for frozen semantic-vocabulary BEIR query selection."""

from __future__ import annotations

import json
from types import SimpleNamespace

from benchmarks.fitz_bench.beir_holdout import (
    build_holdout_manifest,
    load_query_manifest,
    manifest_query_selection,
    query_manifest_digest,
)


def test_holdout_selection_is_balanced_deterministic_and_score_blind(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "test.tsv"
    corpus_records = []
    query_records = []
    qrel_lines = ["query-id\tcorpus-id\tscore"]
    document_texts = [
        "delta epsilon",
        "delta epsilon",
        "delta epsilon",
        "alpha delta epsilon",
        "alpha delta epsilon",
        "alpha delta epsilon",
        "alpha beta delta",
        "alpha beta delta",
        "alpha beta delta",
    ]
    for index, document_text in enumerate(document_texts):
        corpus_records.append(json.dumps({"_id": f"d{index}", "title": "", "text": document_text}))
        query_records.append(json.dumps({"_id": f"q{index}", "text": "alpha beta gamma"}))
        qrel_lines.append(f"q{index}\td{index}\t1")
    query_records.append(json.dumps({"_id": "q9", "text": "missing positive"}))
    qrel_lines.append("q9\td-missing\t1")
    corpus.write_text("\n".join(corpus_records) + "\n", encoding="utf-8")
    queries.write_text("\n".join(query_records) + "\n", encoding="utf-8")
    qrels.write_text("\n".join(qrel_lines) + "\n", encoding="utf-8")
    dataset = SimpleNamespace(
        name="toy",
        md5="archive-md5",
        source_corpus=str(corpus),
        source_queries=str(queries),
        source_qrels=str(qrels),
        corpus_documents=9,
        ignore_identical_ids=True,
    )

    first = build_holdout_manifest([dataset], sample_size=6, seed=17)
    second = build_holdout_manifest([dataset], sample_size=6, seed=17)

    assert first == second
    assert first["datasets"]["toy"]["group_counts"] == {
        "low": 2,
        "medium": 2,
        "high": 2,
    }
    assert len(first["datasets"]["toy"]["queries"]) == 6
    assert first["datasets"]["toy"]["available_judged_queries"] == 10
    assert first["datasets"]["toy"]["eligible_queries"] == 9
    assert first["datasets"]["toy"]["missing_positive_qrels"] == [
        {"query_id": "q9", "document_id": "d-missing"}
    ]
    assert first["datasets"]["toy"]["queries_without_available_positives"] == ["q9"]
    assert query_manifest_digest(first) == query_manifest_digest(second)


def test_manifest_round_trip_and_dataset_selection(tmp_path) -> None:
    manifest = {
        "schema_version": 1,
        "name": "beir-semantic-vocabulary-holdout-v1",
        "selection": {
            "algorithm": "positive-max-token-jaccard-tertiles-sha256-v1",
        },
        "datasets": {
            "toy": {
                "archive_md5": "archive-md5",
                "queries": [
                    {
                        "query_id": "q1",
                        "group": "low",
                        "positive_max_token_jaccard": 0.1,
                    }
                ],
            }
        },
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_query_manifest(path)
    query_ids, metadata, summary = manifest_query_selection(
        loaded,
        SimpleNamespace(name="toy", md5="archive-md5"),
        available_query_ids={"q1"},
    )

    assert query_ids == ["q1"]
    assert metadata == {
        "q1": {
            "group": "low",
            "positive_max_token_jaccard": 0.1,
        }
    }
    assert summary["group_counts"] == {"low": 1}
    assert summary["manifest_sha256"] == query_manifest_digest(loaded)
