# tests/unit/test_retrieval_wiring.py
"""
Tests for retrieval pipeline wiring: router dispatch logic, keyword enrichment
boost, and freshness boost in section/code strategies.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_sage.engines.fitz_krag.retrieval.strategies.code_search import CodeSearchStrategy
from fitz_sage.engines.fitz_krag.retrieval.strategies.section_search import (
    SectionSearchStrategy,
)
from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> MagicMock:
    """Build a mock FitzKragConfig with sensible defaults."""
    cfg = MagicMock()
    cfg.top_addresses = overrides.get("top_addresses", 10)
    cfg.enable_multi_query = overrides.get("enable_multi_query", False)
    cfg.multi_query_min_length = overrides.get("multi_query_min_length", 300)
    cfg.min_relevance_score = overrides.get("min_relevance_score", 0)
    cfg.keyword_weight = overrides.get("keyword_weight", 0.4)
    cfg.code_bm25_weight = overrides.get("code_bm25_weight", 0.3)
    return cfg


def _make_router(
    config=None,
    code_strategy=None,
    section_strategy=None,
    table_strategy=None,
    chat_factory=None,
) -> RetrievalRouter:
    """Build a RetrievalRouter with mocked strategies."""
    config = config or _make_config()
    code = code_strategy or MagicMock()
    code.retrieve = code.retrieve if code_strategy else MagicMock(return_value=[])

    router = RetrievalRouter(
        code_strategy=code,
        config=config,
        section_strategy=section_strategy,
        table_strategy=table_strategy,
        chat_factory=chat_factory,
    )
    return router


def _make_address(source_id: str = "f1", location: str = "loc", score: float = 0.5) -> Address:
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary="summary",
        score=score,
        metadata={},
    )


def _make_section_address(
    source_id: str = "f1", location: str = "sec", score: float = 0.5
) -> Address:
    return Address(
        kind=AddressKind.SECTION,
        source_id=source_id,
        location=location,
        summary="summary",
        score=score,
        metadata={},
    )


# ===========================================================================
# Router tests
# ===========================================================================


class TestRouterRewriteVariations:
    """Test 1: Router uses rewriter's all_query_variations as additional queries."""

    def test_router_uses_rewrite_variations(self):
        code_strategy = MagicMock()
        code_strategy.retrieve = MagicMock(return_value=[])
        section_strategy = MagicMock()
        section_strategy.retrieve = MagicMock(return_value=[])

        router = _make_router(
            code_strategy=code_strategy,
            section_strategy=section_strategy,
        )

        rewrite_result = SimpleNamespace(all_query_variations=["q1", "q2", "q3"])

        router.retrieve("q1", rewrite_result=rewrite_result)

        # Strategies should be called with original query "q1" plus variations "q2" and "q3"
        code_calls = [c.args[0] for c in code_strategy.retrieve.call_args_list]
        assert "q1" in code_calls
        assert "q2" in code_calls
        assert "q3" in code_calls
        # 3 total calls: original + 2 variations
        assert code_strategy.retrieve.call_count == 3


class TestRouterFallbackToExpandQuery:
    """Test 2: Without rewrite_result, router falls back to _expand_query when profile.run_multi_query=True."""

    def test_router_falls_back_to_expand_query(self):
        code_strategy = MagicMock()
        code_strategy.retrieve = MagicMock(return_value=[])

        router = _make_router(code_strategy=code_strategy)

        long_query = "a" * 300

        profile = RetrievalProfile(run_multi_query=True)

        with patch.object(router, "_expand_query", return_value=["eq1", "eq2"]) as mock_expand:
            router.retrieve(long_query, profile)
            mock_expand.assert_called_once_with(long_query)

        # Original query + 2 expanded = 3 calls
        assert code_strategy.retrieve.call_count == 3


class TestRouterSkipsExpandWhenRewriteHasVariations:
    """Test 3: _expand_query NOT called when rewrite_result has variations."""

    def test_router_skips_expand_when_rewrite_has_variations(self):
        chat_factory = MagicMock()
        config = _make_config(enable_multi_query=True, multi_query_min_length=5)

        code_strategy = MagicMock()
        code_strategy.retrieve = MagicMock(return_value=[])

        router = _make_router(
            config=config,
            code_strategy=code_strategy,
            chat_factory=chat_factory,
        )

        rewrite_result = SimpleNamespace(all_query_variations=["q1", "q2", "q3"])

        with patch.object(router, "_expand_query", return_value=["eq1"]) as mock_expand:
            router.retrieve("q1", rewrite_result=rewrite_result)
            mock_expand.assert_not_called()


class TestRouterUsesDetectionComparisonQueries:
    """Test 4: Router uses comparison_queries from RetrievalProfile."""

    def test_router_uses_detection_comparison_queries(self):
        code_strategy = MagicMock()
        code_strategy.retrieve = MagicMock(return_value=[])

        router = _make_router(code_strategy=code_strategy)

        profile = RetrievalProfile(comparison_queries=["cq1", "cq2"])

        router.retrieve("original query", profile)

        queries_called = [c.args[0] for c in code_strategy.retrieve.call_args_list]
        assert "cq1" in queries_called
        assert "cq2" in queries_called


class TestRouterEntityFallbackWithoutComparisonQueries:
    """Test 5: profile has comparison_entities but no comparison_queries — entity fallback."""

    def test_router_entity_fallback_without_comparison_queries(self):
        code_strategy = MagicMock()
        code_strategy.retrieve = MagicMock(return_value=[])

        router = _make_router(code_strategy=code_strategy)

        profile = RetrievalProfile(
            comparison_queries=[],  # Empty — triggers entity fallback
            comparison_entities=["EntityA", "EntityB"],
        )

        router.retrieve("how does", profile)

        queries_called = [c.args[0] for c in code_strategy.retrieve.call_args_list]
        assert "how does EntityA" in queries_called
        assert "how does EntityB" in queries_called


class TestRouterPassesDetectionToStrategies:
    """Test 6: profile flows to strategy.retrieve() as detection kwarg."""

    def test_router_passes_detection_to_strategies(self):
        code_strategy = MagicMock()
        code_strategy.retrieve = MagicMock(return_value=[])
        section_strategy = MagicMock()
        section_strategy.retrieve = MagicMock(return_value=[])
        table_strategy = MagicMock()
        table_strategy.retrieve = MagicMock(return_value=[])

        router = _make_router(
            code_strategy=code_strategy,
            section_strategy=section_strategy,
            table_strategy=table_strategy,
        )

        profile = RetrievalProfile()

        router.retrieve("test query", profile)

        # Every strategy call should receive detection=profile
        for call in code_strategy.retrieve.call_args_list:
            assert call.kwargs.get("detection") is profile

        for call in section_strategy.retrieve.call_args_list:
            assert call.kwargs.get("detection") is profile

        for call in table_strategy.retrieve.call_args_list:
            assert call.kwargs.get("detection") is profile


# ===========================================================================
# Section strategy tests
# ===========================================================================


def _make_section_strategy(
    section_store=None, config=None, raw_store=None
) -> SectionSearchStrategy:
    """Build a SectionSearchStrategy with mocked dependencies."""
    store = section_store or MagicMock()
    cfg = config or _make_config()
    strategy = SectionSearchStrategy(store, cfg)
    strategy._raw_store = raw_store
    return strategy


class TestSectionKeywordBoostIncreasesScore:
    """Test 7: keyword enrichment boost increases combined_score."""

    def test_section_keyword_boost_increases_score(self):
        section_store = MagicMock()
        # BM25 returns one result
        section_store.search_bm25.return_value = [
            {
                "id": "s1",
                "raw_file_id": "f1",
                "title": "Setup",
                "summary": "setup guide",
                "level": 1,
                "bm25_score": 0.5,
            }
        ]
        # Semantic returns empty
        section_store.search_by_vector.return_value = []
        # Keyword enrichment returns a hit for s1
        section_store.search_by_keywords.return_value = [{"id": "s1"}]

        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        strategy = _make_section_strategy(section_store=section_store)
        results = strategy.retrieve("setup guide test", limit=5)

        # Score should include the 0.1 keyword boost on top of BM25 contribution
        assert len(results) == 1
        # RRF with k=60: BM25 only at rank 0 → 1/(60+0) = 1/60 ≈ 0.0167
        # With keyword boost: 1/60 + 0.1 ≈ 0.1167
        assert results[0].score == pytest.approx(1 / 60 + 0.1, abs=0.01)


# ===========================================================================
# Code strategy tests
# ===========================================================================


def _make_code_strategy(
    symbol_store=None, config=None, raw_store=None
) -> CodeSearchStrategy:
    """Build a CodeSearchStrategy with mocked dependencies."""
    store = symbol_store or MagicMock()
    cfg = config or _make_config()
    strategy = CodeSearchStrategy(store, cfg)
    strategy._raw_store = raw_store
    return strategy


class TestCodeKeywordBoostIncreasesScore:
    """Test 8: keyword enrichment boost increases combined_score for code."""

    def test_code_keyword_boost_increases_score(self):
        symbol_store = MagicMock()
        # Keyword search by name returns one result
        symbol_store.search_by_name.return_value = [
            {
                "id": "sym1",
                "raw_file_id": "f1",
                "name": "parse_config",
                "qualified_name": "mod.parse_config",
                "kind": "function",
                "start_line": 10,
                "end_line": 30,
                "summary": "parses config",
            }
        ]
        # BM25 returns empty
        symbol_store.search_bm25.return_value = []
        # Semantic returns empty
        symbol_store.search_by_vector.return_value = []
        # Keyword enrichment returns a hit
        symbol_store.search_by_keywords.return_value = [{"id": "sym1"}]

        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        strategy = _make_code_strategy(symbol_store=symbol_store)
        results = strategy.retrieve("parse config file", limit=5)

        assert len(results) == 1
        # Keyword rank score alone: keyword_weight * (1/(0+1)) = 0.4 * 1.0 = 0.4
        # With keyword enrichment boost: 0.4 + 0.1 = 0.5
        assert results[0].score == pytest.approx(0.5, abs=0.01)


class TestKeywordBoostSkipsShortTerms:
    """Test 9: query with only short words (<3 chars) gets no keyword boost."""

    def test_keyword_boost_skips_short_terms(self):
        section_store = MagicMock()
        section_store.search_bm25.return_value = [
            {
                "id": "s1",
                "raw_file_id": "f1",
                "title": "A",
                "summary": "a",
                "level": 1,
                "bm25_score": 0.5,
            }
        ]
        section_store.search_by_vector.return_value = []

        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        strategy = _make_section_strategy(section_store=section_store)
        # All words are shorter than 3 chars
        results = strategy.retrieve("is a do", limit=5)

        # search_by_keywords should NOT be called because all terms are < 3 chars
        section_store.search_by_keywords.assert_not_called()
        assert len(results) == 1


class TestSectionFreshnessBoostWithRecency:
    """Test 10: freshness boost adds score to files with recent timestamps."""

    def test_section_freshness_boost_with_recency(self):
        section_store = MagicMock()
        # Return results from 4 different files so top_quarter / top_half logic works
        section_store.search_bm25.return_value = [
            {
                "id": f"s{i}",
                "raw_file_id": f"f{i}",
                "title": f"Section {i}",
                "summary": f"summary {i}",
                "level": 1,
                "bm25_score": 0.5,
            }
            for i in range(4)
        ]
        section_store.search_by_vector.return_value = []
        section_store.search_by_keywords.return_value = []

        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        raw_store = MagicMock()
        # f0 is most recent, f3 is oldest
        raw_store.get_updated_timestamps.return_value = {
            "f0": "2025-05-01",
            "f1": "2025-04-01",
            "f2": "2025-03-01",
            "f3": "2025-02-01",
        }

        strategy = _make_section_strategy(
            section_store=section_store,
            raw_store=raw_store,
        )

        detection = SimpleNamespace(boost_recency=True)
        results = strategy.retrieve("recent changes", limit=10, detection=detection)

        # With 4 files: top_quarter = f0, top_half = f0,f1
        # f0 gets +0.1, f1 gets +0.05, f2 and f3 get nothing
        scores = {r.source_id: r.score for r in results}
        # RRF with k=60: BM25 only at ranks 0,1,2,3 → 1/(60+rank)
        assert scores["f0"] > scores["f2"]
        assert scores["f0"] > scores["f3"]
        assert scores["f0"] == pytest.approx(1 / 60 + 0.1, abs=0.01)
        assert scores["f1"] == pytest.approx(1 / 61 + 0.05, abs=0.01)
        assert scores["f2"] == pytest.approx(1 / 62, abs=0.001)


class TestFreshnessNoBoostWithoutFlag:
    """Test 11: detection.boost_recency=False means no raw_store call."""

    def test_freshness_no_boost_without_flag(self):
        section_store = MagicMock()
        section_store.search_bm25.return_value = [
            {
                "id": "s1",
                "raw_file_id": "f1",
                "title": "T",
                "summary": "s",
                "level": 1,
                "bm25_score": 0.5,
            }
        ]
        section_store.search_by_vector.return_value = []
        section_store.search_by_keywords.return_value = []

        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        raw_store = MagicMock()

        strategy = _make_section_strategy(
            section_store=section_store,
            raw_store=raw_store,
        )

        detection = SimpleNamespace(boost_recency=False)
        strategy.retrieve("anything", limit=5, detection=detection)

        raw_store.get_updated_timestamps.assert_not_called()


class TestFreshnessNoBoostWithoutRawStore:
    """Test 12: detection.boost_recency=True but _raw_store=None -> no crash."""

    def test_freshness_no_boost_without_raw_store(self):
        section_store = MagicMock()
        section_store.search_bm25.return_value = [
            {
                "id": "s1",
                "raw_file_id": "f1",
                "title": "T",
                "summary": "s",
                "level": 1,
                "bm25_score": 0.5,
            }
        ]
        section_store.search_by_vector.return_value = []
        section_store.search_by_keywords.return_value = []

        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        strategy = _make_section_strategy(
            section_store=section_store,
            raw_store=None,  # Explicitly None
        )

        detection = SimpleNamespace(boost_recency=True)

        # Should not crash
        results = strategy.retrieve("anything", limit=5, detection=detection)
        assert len(results) == 1
