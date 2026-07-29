"""Focused tests for query-pipeline boundaries."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fitz_sage.core.exceptions import QueryError
from fitz_sage.engines.fitz_krag.evidence_closure import EvidenceClosureRequest
from fitz_sage.engines.fitz_krag.query_pipeline import (
    _closure_profile,
    _filter_companion_source_repeats,
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

    assert _filter_companion_source_repeats(
        _section_request(),
        [glossary],
        [repeated],
    ) == []


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
