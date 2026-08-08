"""Tests for the verified BEIR corpus adapter."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from benchmarks.fitz_bench import beir
from benchmarks.fitz_bench.beir import (
    DatasetSpec,
    load_mapping,
    load_qrels,
    prepare_dataset,
    projected_content,
)


def test_prepare_dataset_projects_exact_title_and_text_fields(
    monkeypatch,
    tmp_path,
) -> None:
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "toy.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(
            "toy/corpus.jsonl",
            json.dumps(
                {
                    "_id": "unsafe/id:1",
                    "title": "  Exact title  ",
                    "text": "Exact body\nsecond line",
                    "metadata": {"must_not_be_indexed": "secret marker"},
                }
            )
            + "\n",
        )
        output.writestr(
            "toy/queries.jsonl",
            json.dumps({"_id": "q1", "text": "Exact body"}) + "\n",
        )
        output.writestr(
            "toy/qrels/test.tsv",
            "query-id\tcorpus-id\tscore\nq1\tunsafe/id:1\t2\n",
        )
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    monkeypatch.setitem(
        beir.DATASETS,
        "toy",
        DatasetSpec(name="toy", md5=digest, corpus_documents=1, test_queries=1),
    )

    prepared = prepare_dataset(tmp_path, "toy", offline=True)

    by_path, by_document, hashes_by_path = load_mapping(Path(prepared.mapping_path))
    projected_path = Path(prepared.corpus_dir) / by_document["unsafe/id:1"]
    assert projected_path.read_text(encoding="utf-8") == (
        "  Exact title  \n\nExact body\nsecond line"
    )
    assert "secret marker" not in projected_path.read_text(encoding="utf-8")
    assert by_path[by_document["unsafe/id:1"]] == "unsafe/id:1"
    assert (
        hashes_by_path[by_document["unsafe/id:1"]]
        == hashlib.sha256(projected_path.read_bytes()).hexdigest()
    )
    assert prepared.corpus_documents == 1
    assert prepared.empty_documents == 0
    assert prepared.empty_judged_relevant_documents == 0
    assert prepared.test_queries == 1
    assert prepared.qrels == 1

    Path(prepared.source_corpus).write_text(
        json.dumps({"_id": "unsafe/id:1", "title": "changed", "text": "changed"}) + "\n",
        encoding="utf-8",
    )
    restored = prepare_dataset(tmp_path, "toy", offline=True)
    assert "changed" not in Path(restored.source_corpus).read_text(encoding="utf-8")


def test_projected_content_does_not_strip_or_invent_empty_fields() -> None:
    assert projected_content({"title": "", "text": " body "}) == " body "
    assert projected_content({"title": " title ", "text": ""}) == " title "
    assert projected_content({"title": "", "text": ""}) == ""


def test_load_qrels_preserves_graded_relevance(tmp_path) -> None:
    path = tmp_path / "test.tsv"
    path.write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\n",
        encoding="utf-8",
    )

    assert load_qrels(path) == {"q1": {"d1": 2, "d2": 0}}


def test_supported_specs_use_published_archive_contract() -> None:
    assert beir.DATASETS["scifact"].url.endswith("/scifact.zip")
    assert beir.DATASETS["scifact"].md5 == "5f7d1de60b170fc8027bb7898e2efca1"
    assert beir.DATASETS["arguana"].corpus_documents == 8674
    assert beir.DATASETS["arguana"].ignore_identical_ids is True
    assert beir.DATASETS["quora"].corpus_documents == 522931
    assert beir.DATASETS["quora"].ignore_identical_ids is True
