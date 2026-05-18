# tests/unit/test_krag_vocabulary.py
"""
Unit tests for vocabulary integration in the KRAG ingestion core and router.

The ingestion tests exercise the *live* path: the background worker schedules
``core.enrich_file``, which extracts keywords and merges them into the
VocabularyStore. This is what ``engine.point()`` runs in production.

Tests that:
- enrich_file saves extracted keywords to the VocabularyStore
- keywords are deduplicated case-insensitively before saving
- enrich_file is safe when no VocabularyStore is wired
- Router's _apply_keyword_boost boosts matching addresses
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_sage.engines.fitz_krag.types import Address, AddressKind
from fitz_sage.retrieval.vocabulary.models import Keyword

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_core(*, vocabulary_store: MagicMock | None = None):
    """Create a KragIngestPipeline core with a mocked connection manager."""
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

    config = FitzKragConfig(collection="test_col", enable_enrichment=True)
    return KragIngestPipeline(
        config=config,
        chat=MagicMock(),
        connection_manager=MagicMock(),
        collection="test_col",
        vocabulary_store=vocabulary_store,
    )


def _enricher_stamping(keyword_sets: list[list[str]]) -> MagicMock:
    """Enricher stub: stamps the i-th keyword set onto the i-th symbol dict."""
    enricher = MagicMock()

    def _stamp(dicts):
        for i, d in enumerate(dicts):
            d["keywords"] = list(keyword_sets[i % len(keyword_sets)])
            d["entities"] = []

    enricher.enrich_symbols.side_effect = _stamp
    enricher.enrich_sections.side_effect = _stamp
    return enricher


# ---------------------------------------------------------------------------
# TestEnrichVocabularyIntegration — produced by core.enrich_file
# ---------------------------------------------------------------------------


class TestEnrichVocabularyIntegration:
    """Tests that enrich_file feeds extracted keywords into the VocabularyStore."""

    def test_enrich_file_saves_keywords_to_vocabulary(self):
        """enrich_file on a code file merges its extracted keywords into the store."""
        vocab_store = MagicMock()
        core = _make_core(vocabulary_store=vocab_store)
        core._enricher = _enricher_stamping([["auth", "login"], ["hash"]])
        core._symbol_store = MagicMock()
        core._symbol_store.get_by_file.return_value = [
            {"id": "s1", "keywords": [], "entities": []},
            {"id": "s2", "keywords": [], "entities": []},
        ]

        core.enrich_file("file-1", ".py")

        vocab_store.merge_and_save.assert_called_once()
        keywords = vocab_store.merge_and_save.call_args[0][0]
        # 3 unique keywords: auth, login, hash
        assert {kw.id for kw in keywords} == {"auth", "login", "hash"}

    def test_deduplicates_keywords_case_insensitively(self):
        """Duplicate keywords (case-insensitive) collapse to one before saving."""
        vocab_store = MagicMock()
        core = _make_core(vocabulary_store=vocab_store)
        core._enricher = _enricher_stamping([["Auth", "login"], ["auth", "Login"]])
        core._symbol_store = MagicMock()
        core._symbol_store.get_by_file.return_value = [
            {"id": "s1", "keywords": [], "entities": []},
            {"id": "s2", "keywords": [], "entities": []},
        ]

        core.enrich_file("file-1", ".py")

        keywords = vocab_store.merge_and_save.call_args[0][0]
        # "auth"/"Auth" and "login"/"Login" each collapse to one
        assert len(keywords) == 2

    def test_enrich_file_safe_without_vocabulary_store(self):
        """No VocabularyStore wired: enrich_file still enriches, no save attempted."""
        core = _make_core(vocabulary_store=None)
        core._enricher = _enricher_stamping([["auth"]])
        core._symbol_store = MagicMock()
        core._symbol_store.get_by_file.return_value = [
            {"id": "s1", "keywords": [], "entities": []},
        ]

        # Should not raise
        core.enrich_file("file-1", ".py")

        # Enrichment was persisted even though vocabulary is not wired
        core._symbol_store.update_enrichment_by_file.assert_called_once()


# ---------------------------------------------------------------------------
# TestRouterKeywordBoost
# ---------------------------------------------------------------------------


def _make_router_config(top_addresses: int = 10) -> MagicMock:
    """Create a mock FitzKragConfig for the router."""
    cfg = MagicMock()
    cfg.top_addresses = top_addresses
    cfg.retrieval_workers = 4
    cfg.enable_multi_query = False
    return cfg


def _addr(
    kind: AddressKind = AddressKind.SYMBOL,
    source_id: str = "src",
    location: str = "mod.func",
    summary: str = "does something",
    score: float = 0.5,
) -> Address:
    """Build an Address."""
    return Address(
        kind=kind,
        source_id=source_id,
        location=location,
        summary=summary,
        score=score,
    )


def _make_keyword(kw_str: str) -> Keyword:
    """Create a Keyword object from a string."""
    return Keyword(id=kw_str, category="auto", match=[kw_str])


def _make_keyword_matcher(matched_keywords: list[str] | None = None) -> MagicMock:
    """Create a mock KeywordMatcher with find_in_query returning Keyword objects."""
    matcher = MagicMock()
    if matched_keywords is None:
        matched_keywords = []
    matcher.find_in_query.return_value = [_make_keyword(kw) for kw in matched_keywords]
    return matcher


def _make_router(
    code_addresses: list[Address] | None = None,
    keyword_matcher: MagicMock | None = None,
    top_addresses: int = 10,
) -> RetrievalRouter:
    """Create a RetrievalRouter with mocked strategies."""
    code_strat = MagicMock()
    code_strat.retrieve.return_value = code_addresses or []
    config = _make_router_config(top_addresses=top_addresses)
    router = RetrievalRouter(
        code_strategy=code_strat,
        config=config,
    )
    if keyword_matcher:
        router._keyword_matcher = keyword_matcher
    return router


class TestRouterKeywordBoost:
    """Tests for _apply_keyword_boost in RetrievalRouter."""

    def test_boosts_matching_addresses(self):
        """Addresses with keywords in summary/location get score boost."""
        addresses = [
            _addr(score=0.5, location="auth.login_handler", summary="Handles login"),
            _addr(score=0.5, location="utils.helper", summary="Generic helper"),
        ]
        matcher = _make_keyword_matcher(["login", "auth"])
        router = _make_router(keyword_matcher=matcher)

        boosted = router._apply_keyword_boost("how does login work?", addresses)

        # First address matches both keywords -> +0.2 boost
        assert boosted[0].score == pytest.approx(0.5 + 0.1 * 2)
        # Second address matches neither
        assert boosted[1].score == 0.5

    def test_boost_proportional_to_matches(self):
        """Score boost is 0.1 per matched keyword."""
        addresses = [
            _addr(score=0.5, location="mod.func", summary="auth login handler"),
        ]
        # 3 keywords match in summary
        matcher = _make_keyword_matcher(["auth", "login", "handler"])
        router = _make_router(keyword_matcher=matcher)

        boosted = router._apply_keyword_boost("query", addresses)

        assert boosted[0].score == pytest.approx(0.5 + 0.1 * 3)

    def test_no_boost_when_no_keywords_match(self):
        """Addresses unchanged when no vocabulary keywords match."""
        addresses = [
            _addr(score=0.7, location="mod.func", summary="does something"),
            _addr(score=0.4, location="mod.other", summary="other thing"),
        ]
        matcher = _make_keyword_matcher([])
        router = _make_router(keyword_matcher=matcher)

        boosted = router._apply_keyword_boost("unrelated query", addresses)

        assert boosted[0].score == 0.7
        assert boosted[1].score == 0.4

    def test_no_boost_without_keyword_matcher(self):
        """When _keyword_matcher is None, retrieve skips boosting entirely."""
        addresses = [
            _addr(score=0.5, location="a.func"),
            _addr(score=0.3, location="b.func"),
        ]
        router = _make_router(code_addresses=addresses)
        # _keyword_matcher defaults to None

        result = router.retrieve("query")

        # Results returned sorted by score, no boost applied
        assert result[0].score == 0.5
        assert result[1].score == 0.3

    def test_keyword_boost_integrated_in_retrieve(self):
        """Full retrieve flow applies keyword boost before final sort."""
        addresses = [
            _addr(score=0.3, location="low.func", summary="login auth handler"),
            _addr(score=0.6, location="high.func", summary="unrelated code"),
        ]
        matcher = _make_keyword_matcher(["login", "auth"])
        router = _make_router(code_addresses=addresses, keyword_matcher=matcher)

        result = router.retrieve("login auth")

        # low.func gets +0.2 boost -> 0.5, high.func stays 0.6
        # After sort: high.func (0.6), low.func (0.5)
        assert result[0].location == "high.func"
        assert result[1].location == "low.func"
        assert result[1].score == pytest.approx(0.3 + 0.1 * 2)
