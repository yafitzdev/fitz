# tests/unit/test_krag_vocabulary.py
"""
Unit tests for vocabulary integration in the KRAG pipeline and router.

Tests that:
- Pipeline saves keywords to VocabularyStore after enrichment
- Router's _apply_keyword_boost boosts matching addresses
- Boost is proportional to number of matched keywords
- No boost when no keywords match
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_sage.engines.fitz_krag.types import Address, AddressKind
from fitz_sage.retrieval.vocabulary.models import Keyword

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_router_config(top_addresses: int = 10) -> MagicMock:
    """Create a mock FitzKragConfig for the router."""
    cfg = MagicMock()
    cfg.top_addresses = top_addresses
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
    # Return Keyword objects, matching what the real KeywordMatcher.find_in_query returns
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


# ---------------------------------------------------------------------------
# TestPipelineVocabularyIntegration
# ---------------------------------------------------------------------------


class TestPipelineVocabularyIntegration:
    """Tests that the pipeline saves keywords to VocabularyStore."""

    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.ensure_schema")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.RawFileStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.SymbolStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.ImportGraphStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.SectionStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.TableStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.PythonCodeIngestStrategy")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.TechnicalDocIngestStrategy")
    def test_saves_keywords_after_enrichment(
        self,
        mock_doc_strat,
        mock_py_strat,
        mock_table_store,
        mock_section_store,
        mock_import_store,
        mock_symbol_store,
        mock_raw_store,
        mock_ensure_schema,
    ):
        """Pipeline calls vocabulary_store.merge_and_save with extracted keywords."""
        from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
        from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

        config = FitzKragConfig(collection="test_col", enable_enrichment=False)
        chat = MagicMock()
        embedder = MagicMock()
        embedder.dimensions = 1024
        embedder.embed_batch.return_value = [[0.1] * 1024]
        cm = MagicMock()
        vocab_store = MagicMock()

        pipeline = KragIngestPipeline(
            config=config,
            chat=chat,
            connection_manager=cm,
            collection="test_col",
            vocabulary_store=vocab_store,
        )

        # Simulate that _save_keywords_to_vocabulary is called with symbol dicts
        # containing keywords
        symbol_dicts = [
            {"keywords": ["auth", "login"], "entities": []},
            {"keywords": ["hash"], "entities": []},
        ]
        pipeline._save_keywords_to_vocabulary(symbol_dicts, [])

        vocab_store.merge_and_save.assert_called_once()
        call_args = vocab_store.merge_and_save.call_args
        keywords = call_args[0][0]  # First positional arg
        # 3 unique keywords: auth, login, hash
        assert len(keywords) == 3

    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.ensure_schema")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.RawFileStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.SymbolStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.ImportGraphStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.SectionStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.TableStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.PythonCodeIngestStrategy")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.TechnicalDocIngestStrategy")
    def test_no_save_without_vocabulary_store(
        self,
        mock_doc_strat,
        mock_py_strat,
        mock_table_store,
        mock_section_store,
        mock_import_store,
        mock_symbol_store,
        mock_raw_store,
        mock_ensure_schema,
    ):
        """Pipeline does not attempt vocabulary save when vocabulary_store is None."""
        from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
        from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

        config = FitzKragConfig(collection="test_col", enable_enrichment=False)
        chat = MagicMock()
        embedder = MagicMock()
        embedder.dimensions = 1024
        cm = MagicMock()

        pipeline = KragIngestPipeline(
            config=config,
            chat=chat,
            connection_manager=cm,
            collection="test_col",
            vocabulary_store=None,
        )

        # Confirm _vocabulary_store is None
        assert pipeline._vocabulary_store is None

    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.ensure_schema")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.RawFileStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.SymbolStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.ImportGraphStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.SectionStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.TableStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.PythonCodeIngestStrategy")
    @patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.TechnicalDocIngestStrategy")
    def test_deduplicates_keywords(
        self,
        mock_doc_strat,
        mock_py_strat,
        mock_table_store,
        mock_section_store,
        mock_import_store,
        mock_symbol_store,
        mock_raw_store,
        mock_ensure_schema,
    ):
        """Duplicate keywords (case-insensitive) are deduplicated before saving."""
        from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
        from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

        config = FitzKragConfig(collection="test_col", enable_enrichment=False)
        chat = MagicMock()
        embedder = MagicMock()
        embedder.dimensions = 1024
        cm = MagicMock()
        vocab_store = MagicMock()

        pipeline = KragIngestPipeline(
            config=config,
            chat=chat,
            connection_manager=cm,
            collection="test_col",
            vocabulary_store=vocab_store,
        )

        symbol_dicts = [
            {"keywords": ["Auth", "login"], "entities": []},
            {"keywords": ["auth", "Login"], "entities": []},
        ]
        pipeline._save_keywords_to_vocabulary(symbol_dicts, [])

        call_args = vocab_store.merge_and_save.call_args
        keywords = call_args[0][0]
        # "auth" and "Auth" are same (case-insensitive); "login" and "Login" too
        assert len(keywords) == 2


# ---------------------------------------------------------------------------
# TestRouterKeywordBoost
# ---------------------------------------------------------------------------


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
