"""Tests for query-leg provenance and bounded result coverage."""

from __future__ import annotations

from fitz_sage.engines.fitz_krag.retrieval.query_coverage import (
    ensure_query_coverage,
    merge_retrieval_provenance,
    tag_retrieval_query,
)
from fitz_sage.engines.fitz_krag.types import Address, AddressKind


def _address(
    source_id: str,
    *,
    score: float = 0.5,
    metadata: dict | None = None,
    kind: AddressKind = AddressKind.SECTION,
) -> Address:
    return Address(
        kind=kind,
        source_id=source_id,
        location="result",
        summary=source_id,
        score=score,
        metadata=metadata or {},
    )


def test_tag_and_merge_retrieval_provenance() -> None:
    """Duplicate hits retain every query leg, temporal tag, and best score."""
    first = tag_retrieval_query([_address("doc", score=0.4)], "first clause")[0]
    duplicate = tag_retrieval_query(
        [
            _address(
                "doc",
                score=0.8,
                metadata={"temporal_refs": ["Q2"]},
            )
        ],
        "second clause",
    )[0]

    merged = merge_retrieval_provenance(first, duplicate)

    assert merged.score == 0.8
    assert merged.metadata["retrieval_queries"] == [
        "first clause",
        "second clause",
    ]
    assert merged.metadata["temporal_refs"] == ["Q2"]


def test_coverage_replaces_noise_without_expanding_budget() -> None:
    """Each successful leg gets one slot inside the existing output limit."""
    first = _address("first", metadata={"retrieval_queries": ["first clause"]})
    second = _address("second", metadata={"retrieval_queries": ["second clause"]})
    noise_a = _address("noise-a")
    noise_b = _address("noise-b")

    result = ensure_query_coverage(
        [noise_a, noise_b, first, second],
        [noise_a, noise_b],
        ["first clause", "second clause"],
        limit=2,
    )

    assert len(result) == 2
    assert {address.source_id for address in result} == {"first", "second"}


def test_one_candidate_can_cover_multiple_query_legs() -> None:
    """A shared hit should satisfy both legs without consuming another slot."""
    shared = _address(
        "shared",
        metadata={"retrieval_queries": ["first clause", "second clause"]},
    )
    noise = _address("noise")

    result = ensure_query_coverage(
        [shared, noise],
        [shared, noise],
        ["first clause", "second clause"],
        limit=2,
    )

    assert result == [shared, noise]


def test_coverage_does_not_invent_a_hit_for_an_unsuccessful_leg() -> None:
    """A leg with no candidate leaves the selected list unchanged."""
    first = _address("first", metadata={"retrieval_queries": ["first clause"]})
    noise = _address("noise")

    result = ensure_query_coverage(
        [first, noise],
        [first, noise],
        ["first clause", "missing clause"],
        limit=2,
    )

    assert result == [first, noise]
