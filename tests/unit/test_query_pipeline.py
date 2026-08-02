"""Focused tests for query-pipeline boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fitz_sage.core.exceptions import QueryError
from fitz_sage.engines.fitz_krag import query_pipeline as query_pipeline_module
from fitz_sage.engines.fitz_krag.evidence_closure import (
    EvidenceClosurePlan,
    EvidenceClosureRequest,
)
from fitz_sage.engines.fitz_krag.evidence_compiler import EvidenceCompilation
from fitz_sage.engines.fitz_krag.query_pipeline import (
    QueryPipeline,
    RetrievalOutcome,
    _closure_profile,
    _filter_companion_source_repeats,
    _partition_closure_requests,
    _validated_query_text,
)
from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult


def _section(source_id: str, section_id: str, location: str) -> ReadResult:
    address = Address(
        kind=AddressKind.SECTION,
        source_id=source_id,
        location=location,
        summary=location,
        metadata={"section_id": section_id},
    )
    return ReadResult(
        address=address,
        content=location,
        file_path=source_id,
    )


def _section_request() -> EvidenceClosureRequest:
    return EvidenceClosureRequest(
        query="Cell Balancing Task ownership",
        modality="section",
        role="bridge_definition:cell balancing task",
        reason="bridge_definition",
    )


def test_query_validation_preserves_technical_angle_bracket_syntax() -> None:
    query = "  How does vector<T> implement operator<=>?  "

    assert _validated_query_text(query) == "How does vector<T> implement operator<=>?"


def test_query_validation_rejects_oversized_input_without_truncating() -> None:
    with pytest.raises(QueryError, match="8000-character limit"):
        _validated_query_text("x" * 8001)


def test_closure_allows_a_different_section_from_the_same_source() -> None:
    glossary = _section("manual.md", "glossary", "Glossary")
    ownership = _section("manual.md", "ownership", "Ownership")

    assert _filter_companion_source_repeats(
        _section_request(),
        [glossary],
        [ownership],
    ) == [ownership]


def test_closure_filters_an_exact_section_repeat() -> None:
    glossary = _section("manual.md", "glossary", "Glossary")
    repeated = _section("manual.md", "glossary", "Glossary")

    assert (
        _filter_companion_source_repeats(
            _section_request(),
            [glossary],
            [repeated],
        )
        == []
    )


def test_evidence_closure_uses_a_tighter_rerank_budget() -> None:
    config = SimpleNamespace(
        top_addresses=50,
        top_read=50,
        rerank_candidates=32,
        rerank_k=10,
        rerank_min_addresses=2,
    )
    base = RetrievalProfile(
        top_k=75,
        top_read=65,
        rerank_candidates=48,
    )

    closure = _closure_profile(base, config, _section_request())

    assert closure.top_k == 32
    assert closure.top_read == 12
    assert closure.rerank_candidates == 16


def test_evidence_closure_does_not_inherit_original_recall_legs() -> None:
    config = SimpleNamespace(
        top_addresses=50,
        top_read=50,
        rerank_candidates=32,
        rerank_k=10,
        rerank_min_addresses=2,
    )
    base = RetrievalProfile(
        keywords=["original", "semantic phrase"],
        query_variations=["temporal variation"],
        comparison_queries=["comparison leg"],
        comparison_entities=["Alpha", "Beta"],
        temporal_references=["last quarter"],
    )

    closure = _closure_profile(base, config, _section_request())

    assert closure.keywords == []
    assert closure.query_variations == []
    assert closure.comparison_queries == []
    assert closure.comparison_entities == []
    assert closure.temporal_references == []
    assert closure.entities == ("Cell", "Balancing", "Task", "ownership")


@pytest.mark.parametrize(
    ("modality", "expected_weights"),
    [
        ("section", {"code": 0.01, "section": 1.0, "table": 0.01}),
        ("table", {"code": 0.01, "section": 0.01, "table": 1.0}),
        ("symbol", {"code": 1.0, "section": 0.01, "table": 0.01}),
    ],
)
def test_evidence_closure_runs_only_the_requested_strategy(
    modality: str,
    expected_weights: dict[str, float],
) -> None:
    config = SimpleNamespace(
        top_addresses=50,
        top_read=50,
        rerank_candidates=32,
        rerank_k=10,
        rerank_min_addresses=2,
    )
    request = EvidenceClosureRequest(
        query="bridge query",
        modality=modality,
        role=f"required_{modality}",
        reason="missing_required_modality",
    )

    closure = _closure_profile(RetrievalProfile(), config, request)

    assert closure.strategy_weights == expected_weights


def test_evidence_closure_skips_unavailable_physical_modalities() -> None:
    section = _section_request()
    table = EvidenceClosureRequest(
        query="table PAY-209",
        modality="table",
        role="required_table",
        reason="missing_required_modality",
    )

    executable, skipped = _partition_closure_requests(
        [section, table],
        {"section"},
    )

    assert executable == [section]
    assert skipped == [
        {
            "request": {
                "query": "table PAY-209",
                "modality": "table",
                "role": "required_table",
                "reason": "missing_required_modality",
                "bridges": [],
            },
            "reason": "modality_unavailable",
        }
    ]


def test_close_evidence_does_not_execute_an_unavailable_request(monkeypatch) -> None:
    request = EvidenceClosureRequest(
        query="table PAY-209",
        modality="table",
        role="required_table",
        reason="missing_required_modality",
    )
    monkeypatch.setattr(
        query_pipeline_module,
        "plan_evidence_closure",
        lambda *_args, **_kwargs: EvidenceClosurePlan(
            requests=[request],
            metadata={"request_count": 1},
        ),
    )
    pipeline = object.__new__(QueryPipeline)
    pipeline._available_modalities = lambda: {"section"}
    pipeline._retrieval_pass = MagicMock()
    outcome = RetrievalOutcome(
        sanitized="Find PAY-209",
        expanded=[],
        addresses=[],
        timings=[],
        profile=RetrievalProfile(required_modalities=("table",)),
    )

    result = pipeline.close_evidence(outcome, EvidenceCompilation([]))

    pipeline._retrieval_pass.run.assert_not_called()
    trace = result.retrieval_trace["evidence_closure"]
    assert trace["executed_request_count"] == 0
    assert trace["added"] == 0
    assert trace["replaced"] == 0
    assert trace["skipped_requests"][0]["reason"] == "modality_unavailable"
