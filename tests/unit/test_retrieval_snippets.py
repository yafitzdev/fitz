"""Tests for bounded retrieval-time source excerpts."""

from fitz_sage.engines.fitz_krag.retrieval.snippets import (
    _query_terms,
    query_relevant_excerpt,
)


def test_query_relevant_excerpt_finds_a_rare_late_literal() -> None:
    target = "RUN_77 retained seal is KAPPA-END."
    source = ("Routine observations continue. " * 200) + target

    excerpt = query_relevant_excerpt(
        "What retained seal applies to RUN_77?",
        source,
        max_chars=600,
    )

    assert target in excerpt
    assert len(excerpt) <= 600


def test_query_relevant_excerpt_does_not_prefer_a_late_equal_match() -> None:
    source = f"record EARLY {'x' * 400} record LATE {'y' * 400}"

    excerpt = query_relevant_excerpt("record", source, max_chars=200)

    assert "record EARLY" in excerpt
    assert "record LATE" not in excerpt
    assert len(excerpt) <= 200


def test_query_terms_are_bounded_and_retain_specific_identifiers() -> None:
    query = " ".join([*(f"ordinaryterm{index}" for index in range(100)), "RUN_77"])

    terms = _query_terms(query)

    assert len(terms) == 32
    assert "run_77" in terms
