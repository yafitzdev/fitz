"""Tests for versioned retrieval execution records and Pyrrho replay."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fitz_sage.core import (
    CandidateReference,
    CandidateStage,
    EvidenceItem,
    EvidencePack,
    FrozenEvidence,
    PyrrhoExecution,
    PyrrhoReplay,
    QueryExecution,
    QueryTerm,
    RetrievalRun,
    RunEnvironment,
    StrategyExecution,
)
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.runtime import load_retrieval_run, replay_pyrrho


def _sample_run() -> RetrievalRun:
    content = "AX_156 failed during the thermal cycle."
    evidence = EvidencePack(
        query="Which case failed?",
        mode=AnswerMode.INSUFFICIENT,
        items=[
            EvidenceItem(
                rank=1,
                source_id="doc-1",
                file_path="reports/run.txt",
                address_kind="section",
                address_location="Failure",
                line_range=(10, 12),
                score=0.91,
                excerpt=content,
                content=content,
                metadata={
                    "source_summary": "PRIVATE SOURCE SUMMARY",
                    "evidence_compiler": {
                        "rank": 1,
                        "alignment_score": 4,
                        "min_sources": 1,
                        "roles": ["bridge:PRIVATE_BRIDGE"],
                    },
                },
            )
        ],
        reasons=["Need another source."],
        timings={"Retrieval": 0.1},
        indexing_status={"complete": True},
        metadata={
            "engine": "fitz_krag",
            "retrieval_trace": {"summary": "PRIVATE TRACE SUMMARY"},
        },
    )
    return RetrievalRun(
        run_id="run-1",
        created_at="2026-07-26T10:00:00Z",
        query=QueryExecution(
            source_text="Which case failed?",
            sanitized_text="Which case failed?",
            retrieval_text="case failed",
            query_shape="narrow",
            terms=(
                QueryTerm(text="case", origin="literal"),
                QueryTerm(text="failure", origin="semantic"),
            ),
        ),
        evidence=evidence,
        strategies=(
            StrategyExecution(
                strategy="section",
                query="case failed",
                result_count=1,
            ),
        ),
        candidate_stages=(
            CandidateStage(
                name="final",
                candidates=(
                    CandidateReference(
                        rank=1,
                        kind="section",
                        source_id="doc-1",
                        location="Failure",
                        score=0.91,
                    ),
                ),
            ),
        ),
        pyrrho=PyrrhoExecution(
            verdict="INSUFFICIENT",
            evidence_count=1,
            reasons=("Need another source.",),
            decision={
                "schema_version": 1,
                "verdict": "INSUFFICIENT",
                "reason": "Need another source.",
            },
        ),
        ranked_evidence=(
            FrozenEvidence.create(
                rank=1,
                source_id="doc-1",
                file_path="reports/run.txt",
                address_kind="section",
                address_location="Failure",
                address_summary="PRIVATE ADDRESS SUMMARY",
                line_range=(10, 12),
                score=0.91,
                content=content,
                compiler_metadata={
                    "rank": 1,
                    "alignment_score": 4,
                    "min_sources": 1,
                    "roles": ["bridge:PRIVATE_BRIDGE"],
                },
            ),
        ),
        pyrrho_evidence=(
            FrozenEvidence.create(
                rank=1,
                source_id="doc-1",
                file_path="reports/run.txt",
                address_kind="section",
                address_location="Failure",
                address_summary="PRIVATE ADDRESS SUMMARY",
                line_range=(10, 12),
                score=0.91,
                content=content,
                compiler_metadata={
                    "rank": 1,
                    "alignment_score": 4,
                    "min_sources": 1,
                    "roles": ["bridge:PRIVATE_BRIDGE"],
                },
            ),
        ),
        environment=RunEnvironment(
            fitz_sage_version="0.15.0",
            engine="fitz_krag",
            collection="reports",
            config_sha256="config-sha",
            collection_sha256="collection-sha",
            components={"pyrrho": "pyrrho/example"},
            indexing_status={"complete": True},
        ),
    )


class _SufficientPyrrho:
    def decide(self, query, evidence):
        return SimpleNamespace(
            verdict="SUFFICIENT",
            reasons=("Enough evidence.",),
            to_dict=lambda: {
                "schema_version": 1,
                "verdict": "SUFFICIENT",
                "reason": "Enough evidence.",
                "probabilities": {
                    "INSUFFICIENT": 0.01,
                    "DISPUTED": 0.01,
                    "SUFFICIENT": 0.98,
                },
            },
        )


def test_redacted_serialization_removes_all_source_derived_text():
    payload = _sample_run().to_dict()
    serialized = json.dumps(payload)

    assert payload["content_included"] is False
    assert payload["evidence"]["items"][0]["content"] == ""
    assert payload["evidence"]["items"][0]["excerpt"] == ""
    assert payload["evidence"]["items"][0]["metadata"] == {
        "evidence_compiler": {
            "rank": 1,
            "alignment_score": 4,
            "min_sources": 1,
        }
    }
    assert payload["ranked_evidence"][0]["compiler_metadata"] == {
        "rank": 1,
        "alignment_score": 4,
        "min_sources": 1,
    }
    assert "content" not in payload["pyrrho_evidence"][0]
    assert payload["evidence"]["metadata"] == {"engine": "fitz_krag"}
    assert "AX_156 failed" not in serialized
    assert "PRIVATE SOURCE SUMMARY" not in serialized
    assert "PRIVATE TRACE SUMMARY" not in serialized
    assert "PRIVATE ADDRESS SUMMARY" not in serialized
    assert "PRIVATE_BRIDGE" not in serialized


def test_content_bearing_round_trip_is_replayable(tmp_path):
    trace_path = tmp_path / "run.json"
    _sample_run().write(trace_path, include_content=True)

    loaded = RetrievalRun.read(trace_path)

    assert loaded.content_included is True
    assert loaded.ranked_evidence[0].verify_content()
    assert loaded.pyrrho_evidence[0].verify_content()
    assert loaded.ranked_evidence[0].content == "AX_156 failed during the thermal cycle."
    assert loaded.evidence.items[0].content == loaded.ranked_evidence[0].content


def test_content_integrity_failure_is_rejected():
    payload = _sample_run().to_dict(include_content=True)
    payload["ranked_evidence"][0]["content"] = "tampered"

    with pytest.raises(ValueError, match="integrity"):
        RetrievalRun.from_dict(payload)


def test_unsupported_schema_major_is_rejected():
    payload = _sample_run().to_dict()
    payload["schema_version"] = "3.0"

    with pytest.raises(ValueError, match="Unsupported retrieval-run schema"):
        RetrievalRun.from_dict(payload)


def test_missing_record_identity_is_rejected():
    payload = _sample_run().to_dict()
    payload.pop("run_id")

    with pytest.raises(ValueError, match="run_id"):
        RetrievalRun.from_dict(payload)


def test_redacted_trace_cannot_be_replayed(tmp_path):
    trace_path = tmp_path / "redacted.json"
    _sample_run().write(trace_path)

    with pytest.raises(ValueError, match="requires a trace exported with source content"):
        replay_pyrrho(trace_path, _SufficientPyrrho())


def test_pyrrho_replay_uses_exact_delivered_evidence_without_retrieval():
    replay = replay_pyrrho(_sample_run(), _SufficientPyrrho())

    assert replay.source_run_id == "run-1"
    assert replay.source_fitz_sage_version == "0.15.0"
    assert replay.replay_fitz_sage_version
    assert replay.original.verdict == "INSUFFICIENT"
    assert replay.replayed.verdict == "SUFFICIENT"
    assert replay.replayed.evidence_count == 1
    assert replay.evidence.items[0].content.startswith("AX_156")
    assert replay.to_dict()["changed"] is True


def test_pyrrho_replay_round_trip_preserves_versioned_result():
    replay = replay_pyrrho(_sample_run(), _SufficientPyrrho())

    loaded = PyrrhoReplay.from_json(replay.to_json(include_content=True))

    assert loaded.source_run_id == "run-1"
    assert loaded.replayed.verdict == "SUFFICIENT"
    assert loaded.evidence.items[0].content.startswith("AX_156")
    assert loaded.content_included is True


def test_pyrrho_replay_rejects_other_engine_records():
    run = _sample_run()
    run.environment = replace(run.environment, engine="custom")

    with pytest.raises(ValueError, match="fitz_krag"):
        replay_pyrrho(run, _SufficientPyrrho())


def test_load_retrieval_run_accepts_json_string():
    loaded = load_retrieval_run(_sample_run().to_json(include_content=True))

    assert loaded.run_id == "run-1"
    assert loaded.content_included is True


def test_explain_is_deterministic_and_content_free():
    explanation = _sample_run().explain()

    assert "Terms (literal): case" in explanation
    assert "Candidates: final=1" in explanation
    assert "Pyrrho: INSUFFICIENT" in explanation
    assert "AX_156 failed" not in explanation
