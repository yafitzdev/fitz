"""Tests for bounded retrieval-time source excerpts."""

from fitz_sage.engines.fitz_krag.retrieval.snippets import query_relevant_excerpt


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


def test_query_relevant_excerpt_samples_repeated_terms_across_document() -> None:
    source = " ".join(f"record filler-{index}" for index in range(500))

    excerpt = query_relevant_excerpt("record", source, max_chars=200)

    assert "record" in excerpt
    assert "filler-499" in excerpt
    assert len(excerpt) <= 200
