"""Tests for the frozen EnterpriseRAG-Bench retrieval harness."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from benchmarks.fitz_bench.enterprise_rag import (
    EnterpriseQuestion,
    EnterpriseRagSpec,
    iter_archive_documents,
    prepare_dataset,
    queries_and_qrels,
)
from benchmarks.fitz_bench.enterprise_rag_benchmark import _grouped_results
from benchmarks.fitz_bench.enterprise_rag_split import (
    build_split_manifest,
    load_split_manifest,
    selection_from_manifest,
    write_frozen_manifest,
)
from benchmarks.fitz_bench.external_retrieval import (
    load_mapping,
    require_reusable_index,
    source_id_mapping,
    summarize_records,
)
from benchmarks.fitz_bench.sqlite_bm25 import SqliteBm25


def test_enterprise_adapter_preserves_files_and_reports_duplicate_ids(tmp_path) -> None:
    cache, spec, document_id = _tiny_release(tmp_path)

    prepared = prepare_dataset(cache, spec=spec, offline=True)

    assert prepared.spec.corpus_documents == 2
    assert prepared.unique_document_ids == 1
    assert prepared.duplicate_document_ids == {
        document_id: (
            f"slack/team/{document_id}__alpha.txt",
            f"slack/team/{document_id}__conflict.txt",
        )
    }
    assert (prepared.corpus_dir / "slack/team" / f"{document_id}__alpha.txt").read_text(
        encoding="utf-8"
    ) == "alpha policy"
    mapping = load_mapping(prepared.mapping_path, allow_duplicate_document_ids=True)
    assert mapping.paths_by_document[document_id] == tuple(
        prepared.duplicate_document_ids[document_id]
    )
    documents = list(iter_archive_documents(prepared))
    assert [item[1] for item in documents] == [document_id, document_id]
    filtered = list(
        iter_archive_documents(
            prepared,
            excluded_relative_paths=(prepared.duplicate_document_ids[document_id][1],),
        )
    )
    assert [item[0] for item in filtered] == [prepared.duplicate_document_ids[document_id][0]]


def test_enterprise_qrels_collapse_only_the_upstream_repeated_official_id(tmp_path) -> None:
    cache, spec, document_id = _tiny_release(tmp_path)
    prepared = prepare_dataset(cache, spec=spec, offline=True)
    from benchmarks.fitz_bench.enterprise_rag import load_questions

    questions = load_questions(prepared.questions_path, spec=spec)
    queries, qrels = queries_and_qrels(questions)

    assert set(queries) == {"q1", "q2"}
    assert qrels == {"q1": {document_id: 1}}


def test_split_is_score_blind_disjoint_and_category_stratified(tmp_path) -> None:
    questions = _split_questions()
    spec = EnterpriseRagSpec(
        release="test",
        archive_url="https://example.invalid/archive.zip",
        archive_sha256="a" * 64,
        archive_bytes=1,
        questions_url="https://example.invalid/questions.jsonl",
        questions_sha256="b" * 64,
        questions_bytes=1,
        archive_files=1,
        extracted_bytes=1,
        corpus_documents=1,
        source_counts={"slack": 1},
        question_counts={"basic": 10, "semantic": 5, "info_not_found": 2},
    )

    first = build_split_manifest(questions, spec=spec, seed=17)
    second = build_split_manifest(questions, spec=spec, seed=17)

    assert first == second
    development = set(first["splits"]["development"]["query_ids"])
    holdout = set(first["splits"]["holdout"]["query_ids"])
    unscored = set(first["splits"]["unscored"]["query_ids"])
    assert len(development) == 5
    assert len(holdout) == 10
    assert len(unscored) == 2
    assert not development & holdout
    assert not development & unscored
    assert not holdout & unscored
    assert first["splits"]["development"]["category_counts"] == {
        "basic": 3,
        "semantic": 2,
    }

    path = tmp_path / "split.json"
    write_frozen_manifest(path, first)
    loaded = load_split_manifest(path, questions, spec=spec)
    query_ids, metadata, report = selection_from_manifest(
        loaded,
        questions,
        "development",
    )
    assert set(query_ids) == development
    assert set(metadata) == development
    assert report["category_counts"] == {"basic": 3, "semantic": 2}


def test_sqlite_bm25_builds_reuses_and_deduplicates_official_ids(tmp_path) -> None:
    database = tmp_path / "plain.sqlite3"
    documents = (
        ("one.txt", "doc-a", "alpha launch policy"),
        ("two.txt", "doc-a", "alpha conflicting note"),
        ("three.txt", "doc-b", "beta customer record"),
    )
    first = SqliteBm25.open_or_build(
        database,
        fingerprint={"corpus": "one"},
        expected_documents=3,
        documents=lambda: iter(documents),
    )
    try:
        assert first.action == "built"
        assert first.search("alpha", top_k=10) == ["doc-a"]
        assert first.search("beta", top_k=1) == ["doc-b"]
    finally:
        first.close()

    reused = SqliteBm25.open_or_build(
        database,
        fingerprint={"corpus": "one"},
        expected_documents=3,
        documents=lambda: (_ for _ in () if False),
    )
    try:
        assert reused.action == "reused_verified"
        assert reused.search("launch", top_k=1) == ["doc-a"]
    finally:
        reused.close()


def test_grouped_results_keeps_category_and_multidocument_views() -> None:
    def record(query_id: str, category: str, expected: int, score: float) -> dict:
        return {
            "query_id": query_id,
            "evaluation": {
                "category": category,
                "source_types": ["slack"],
                "expected_documents": expected,
            },
            "metrics": {
                "baseline": {"Recall@50": score},
                "final": {"Recall@50": score},
            },
            "latency_seconds": {"fitz_sage": 1.0},
            "failure_attribution": "delivered_hit" if score else "recall",
        }

    grouped = _grouped_results(
        [record("q1", "basic", 1, 1.0), record("q2", "completeness", 3, 0.0)]
    )

    assert grouped["category"]["basic"]["queries"] == 1
    assert grouped["document_cardinality"]["single"]["queries"] == 1
    assert grouped["document_cardinality"]["multiple"]["queries"] == 1


def test_mapping_and_reuse_accept_only_an_explicit_expected_failure(tmp_path) -> None:
    mapping = tmp_path / "mapping.jsonl"
    mapping.write_text(
        "\n".join(
            json.dumps(
                {
                    "document_id": document_id,
                    "relative_path": path,
                    "content_sha256": digest,
                }
            )
            for document_id, path, digest in (
                ("doc-a", "source/a.txt", "a" * 64),
                ("doc-b", "source/b.txt", "b" * 64),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    indexed_entry = type("Entry", (), {"file_id": "source-a", "content_hash": "a" * 64})()
    manifest = type(
        "Manifest",
        (),
        {"entries": lambda _self: {"source/a.txt": indexed_entry}},
    )()

    source_ids, summary = source_id_mapping(
        manifest,
        mapping,
        excluded_relative_paths=("source/b.txt",),
    )

    assert source_ids == {"source-a": "doc-a"}
    assert summary["complete"] is True
    assert summary["excluded_paths"] == ["source/b.txt"]

    source = tmp_path / "corpus"
    source.mkdir()
    engine = type(
        "Engine",
        (),
        {
            "_manifest": manifest,
            "_source_dir": source,
            "indexing_status": lambda _self: {
                "query_ready": True,
                "indexed": 1,
                "failed": 1,
                "failed_files": [{"path": "source/b.txt"}],
                "unsupported": 0,
                "enrichment": {"complete": False},
            },
        },
    )()
    assert (
        require_reusable_index(
            engine,
            expected_source=source,
            expected_documents=1,
            index_mode="source",
            expected_failure_paths=("source/b.txt",),
        )
        is manifest
    )


def test_summary_reports_semantic_expansion_outcomes() -> None:
    records = {
        "q1": {
            "metrics": {},
            "recoveries": [],
            "semantic_query_expansion": {"status": "expanded"},
        },
        "q2": {
            "metrics": {},
            "recoveries": [],
            "semantic_query_expansion": {"status": "failed"},
        },
    }

    summary = summarize_records(records)

    assert summary["semantic_query_expansion"] == {"expanded": 1, "failed": 1}


def _tiny_release(tmp_path: Path) -> tuple[Path, EnterpriseRagSpec, str]:
    cache = tmp_path / "cache"
    release = "test-v1"
    release_dir = cache / release
    release_dir.mkdir(parents=True)
    document_id = f"dsid_{'a' * 32}"
    questions = [
        {
            "question_id": "q1",
            "question_type": "basic",
            "source_types": ["slack"],
            "question": "What is the alpha policy?",
            "expected_doc_ids": [document_id, document_id],
        },
        {
            "question_id": "q2",
            "question_type": "info_not_found",
            "source_types": [],
            "question": "What is unavailable?",
            "expected_doc_ids": [],
        },
    ]
    questions_bytes = "".join(json.dumps(question) + "\n" for question in questions).encode("utf-8")
    questions_path = release_dir / "questions.jsonl"
    questions_path.write_bytes(questions_bytes)
    documents = {
        f"slack/team/{document_id}__alpha.txt": b"alpha policy",
        f"slack/team/{document_id}__conflict.txt": b"conflicting policy",
    }
    archive = release_dir / "all_documents.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for path, content in documents.items():
            output.writestr(path, content)
        output.writestr("questions.jsonl", questions_bytes)
    spec = EnterpriseRagSpec(
        release=release,
        archive_url="https://example.invalid/all_documents.zip",
        archive_sha256=_sha256(archive),
        archive_bytes=archive.stat().st_size,
        questions_url="https://example.invalid/questions.jsonl",
        questions_sha256=_sha256(questions_path),
        questions_bytes=questions_path.stat().st_size,
        archive_files=3,
        extracted_bytes=sum(len(value) for value in documents.values()) + len(questions_bytes),
        corpus_documents=2,
        source_counts={"slack": 2},
        question_counts={"basic": 1, "info_not_found": 1},
    )
    return cache, spec, document_id


def _split_questions() -> dict[str, EnterpriseQuestion]:
    questions: dict[str, EnterpriseQuestion] = {}
    for category, count in (("basic", 10), ("semantic", 5)):
        for index in range(count):
            query_id = f"{category}-{index}"
            questions[query_id] = EnterpriseQuestion(
                question_id=query_id,
                question_type=category,
                source_types=("slack",),
                text=f"Question {query_id}",
                expected_document_ids=(f"doc-{query_id}",),
            )
    for index in range(2):
        query_id = f"unscored-{index}"
        questions[query_id] = EnterpriseQuestion(
            question_id=query_id,
            question_type="info_not_found",
            source_types=(),
            text=f"Question {query_id}",
            expected_document_ids=(),
        )
    return questions


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
