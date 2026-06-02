# tests/unit/test_retrieval_pass.py
"""
Unit tests for RetrievalPass — Tiers 1-4 of the retrieval stack
(retrieve -> fuse -> rerank -> read) wired into one composable unit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.retrieval_pass import RetrievalPass
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult


def _addr(
    location: str = "mod.func",
    source_id: str = "src",
    summary: str | None = None,
    score: float = 0.9,
    metadata: dict | None = None,
) -> Address:
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=summary or f"Symbol {location}",
        score=score,
        metadata=metadata or {},
    )


def _read_result(addr: Address) -> ReadResult:
    return ReadResult(address=addr, content="body", file_path="module.py", line_range=(1, 2))


def _build(router_addresses: list[Address], *, with_reranker: bool = True):
    """Construct a RetrievalPass over mocked router / reranker / reader."""
    router = MagicMock(name="router")
    router.retrieve.return_value = router_addresses

    reranker = None
    if with_reranker:
        reranker = MagicMock(name="reranker")
        reranker.rerank.side_effect = lambda query, addrs: addrs  # identity

    reader = MagicMock(name="reader")
    reader.read.side_effect = lambda addrs, limit: [_read_result(a) for a in addrs]

    config = SimpleNamespace(top_read=50)
    return RetrievalPass(router, reranker, reader, config), router, reranker, reader


class TestRetrievalPass:
    """Tiers 1-4 wired into one run()."""

    def test_runs_retrieve_rerank_read(self):
        a1, a2 = _addr("a"), _addr("b")
        rp, router, reranker, reader = _build([a1, a2])

        results = rp.run("query", profile=None)

        router.retrieve.assert_called_once()
        reranker.rerank.assert_called_once()
        reader.read.assert_called_once()
        assert [r.address.location for r in results] == ["a", "b"]

    def test_empty_retrieval_returns_empty(self):
        rp, router, reranker, reader = _build([])

        results = rp.run("query")

        assert results == []
        reranker.rerank.assert_not_called()
        reader.read.assert_not_called()

    def test_exclude_filters_before_rerank(self):
        a1 = _addr(location="a", source_id="s1")
        a2 = _addr(location="b", source_id="s2")
        rp, router, reranker, reader = _build([a1, a2])

        rp.run("query", exclude={("s1", "a")})

        # The excluded address is dropped before it reaches the reranker.
        reranked_input = reranker.rerank.call_args[0][1]
        assert [a.location for a in reranked_input] == ["b"]

    def test_no_reranker_skips_rerank(self):
        a1 = _addr("a")
        rp, router, _reranker, reader = _build([a1], with_reranker=False)

        results = rp.run("query")

        reader.read.assert_called_once()
        assert [r.address.location for r in results] == ["a"]

    def test_broad_query_defers_duplicate_files_after_rerank(self):
        a1 = _addr(location="doc-a-file", source_id="doc-a")
        a2 = _addr(location="doc-a-section", source_id="doc-a")
        b1 = _addr(location="doc-b-file", source_id="doc-b")
        rp, _router, reranker, _reader = _build([a1, a2, b1])
        reranker.rerank.return_value = [a1, a2, b1]
        profile = SimpleNamespace(specificity="broad", answer_type="exploratory")

        results = rp.run("What are the key facts?", profile=profile)

        assert [r.address.location for r in results] == [
            "doc-a-file",
            "doc-b-file",
            "doc-a-section",
        ]

    def test_broad_corpus_query_promotes_overview_files_after_rerank(self):
        test_cases = _addr(
            location="Summary",
            source_id="keyword_test/test_cases.md",
            summary="Sprint 47 test results",
        )
        roadmap = _addr(
            location="product_roadmap_2024.md",
            source_id="product_roadmap_2024.md",
            summary="Product roadmap and launch priorities",
        )
        quarterly = _addr(
            location="quarterly_summary_q2_2024.md",
            source_id="quarterly_summary_q2_2024.md",
            summary="Q2 executive summary and key metrics",
        )
        rp, _router, reranker, _reader = _build([test_cases, roadmap, quarterly])
        reranker.rerank.return_value = [test_cases, roadmap, quarterly]
        profile = SimpleNamespace(specificity="broad", answer_type="exploratory")

        results = rp.run("What are the key facts in this corpus?", profile=profile)

        assert [r.address.source_id for r in results] == [
            "quarterly_summary_q2_2024.md",
            "product_roadmap_2024.md",
            "keyword_test/test_cases.md",
        ]

    def test_broad_corpus_query_rescues_overview_file_dropped_by_reranker(self):
        queries = _addr(
            location="Queries",
            source_id="retrieval_logic/base/queries.md",
            summary="Queries used by retrieval tests",
        )
        q2 = _addr(
            location="quarterly_summary_q2_2024.md",
            source_id="quarterly_summary_q2_2024.md",
            summary="Q2 executive summary and key metrics",
        )
        roadmap = _addr(
            location="product_roadmap_2024.md",
            source_id="product_roadmap_2024.md",
            summary="Product roadmap and launch priorities",
        )
        rp, _router, reranker, _reader = _build([queries, q2, roadmap])
        reranker.rerank.return_value = [queries, q2]
        profile = SimpleNamespace(specificity="broad", answer_type="exploratory")

        results = rp.run("What are the key facts in this corpus?", profile=profile)

        assert [r.address.source_id for r in results] == [
            "quarterly_summary_q2_2024.md",
            "product_roadmap_2024.md",
            "retrieval_logic/base/queries.md",
        ]

    def test_broad_corpus_query_prioritizes_summaries_over_generic_overviews(self):
        feedback = _addr(
            location="Overview > Agent Quality",
            source_id="hierarchical_rag/feedback_march_2024.md",
            summary="Customer feedback overview",
        )
        q1 = _addr(
            location="quarterly_summary_q1_2024.md",
            source_id="quarterly_summary_q1_2024.md",
            summary="Q1 executive summary and key metrics",
        )
        roadmap = _addr(
            location="product_roadmap_2024.md",
            source_id="product_roadmap_2024.md",
            summary="Product roadmap and launch priorities",
        )
        rp, _router, reranker, _reader = _build([feedback, q1, roadmap])
        reranker.rerank.return_value = [feedback, q1, roadmap]
        profile = SimpleNamespace(specificity="broad", answer_type="exploratory")

        results = rp.run("What are the key facts in this corpus?", profile=profile)

        assert [r.address.source_id for r in results] == [
            "quarterly_summary_q1_2024.md",
            "product_roadmap_2024.md",
            "hierarchical_rag/feedback_march_2024.md",
        ]

    def test_broad_corpus_query_does_not_promote_content_summary_words(self):
        incident = _addr(
            location="A16 Incident 17B Impact Summary",
            source_id=(
                "retrieval_logic/near_duplicate_poisoning/artifacts/"
                "A16_incident_17b_impact_summary.txt"
            ),
            summary="Title: Incident Report Summary",
        )
        feedback = _addr(
            location="Overview > Agent Quality",
            source_id="hierarchical_rag/feedback_march_2024.md",
            summary="Customer comments",
        )
        rp, _router, reranker, _reader = _build([incident, feedback])
        reranker.rerank.return_value = [incident, feedback]
        profile = SimpleNamespace(specificity="broad", answer_type="exploratory")

        results = rp.run("What are the key facts in this corpus?", profile=profile)

        assert [r.address.source_id for r in results] == [
            "hierarchical_rag/feedback_march_2024.md",
            (
                "retrieval_logic/near_duplicate_poisoning/artifacts/"
                "A16_incident_17b_impact_summary.txt"
            ),
        ]

    def test_broad_corpus_query_scores_follow_effective_rank(self):
        control = _addr(
            location="Queries",
            source_id="retrieval_logic/base/queries.md",
            summary="Queries used by retrieval tests",
            score=1.5,
        )
        roadmap = _addr(
            location="product_roadmap_2024.md",
            source_id="product_roadmap_2024.md",
            summary="Product roadmap",
            score=0.01,
        )
        rp, _router, reranker, _reader = _build([control, roadmap])
        reranker.rerank.return_value = [control, roadmap]
        profile = SimpleNamespace(specificity="broad", answer_type="exploratory")

        results = rp.run("What are the key facts in this corpus?", profile=profile)

        assert [r.address.source_id for r in results] == [
            "product_roadmap_2024.md",
            "retrieval_logic/base/queries.md",
        ]
        assert results[0].address.score > results[1].address.score
        assert results[0].address.metadata["retrieval_score"] == 0.01
        assert results[1].address.metadata["retrieval_score"] == 1.5
        assert results[0].address.metadata["ranking_score_kind"] == "broad_corpus"

    def test_broad_test_query_keeps_test_surface_order(self):
        test_cases = _addr(
            location="Summary",
            source_id="keyword_test/test_cases.md",
            summary="Sprint 47 test results",
        )
        roadmap = _addr(
            location="product_roadmap_2024.md",
            source_id="product_roadmap_2024.md",
            summary="Product roadmap and launch priorities",
        )
        rp, _router, reranker, _reader = _build([test_cases, roadmap])
        reranker.rerank.return_value = [test_cases, roadmap]
        profile = SimpleNamespace(specificity="broad", answer_type="exploratory")

        results = rp.run("Summarize all test cases", profile=profile)

        assert [r.address.source_id for r in results] == [
            "keyword_test/test_cases.md",
            "product_roadmap_2024.md",
        ]
