# tests/unit/test_chat_only_retrieval.py
"""
Unit tests for retrieval_mode == 'chat_only'.

In chat-only mode the engine doesn't construct an embedder, retrieval
strategies skip semantic search, HyDE is disabled, and the LLMReranker
turns wide BM25 candidates into precise top-k. This is the
philosophy-aligned mode for the honest-RAG thesis.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.retrieval.strategies.chunk_fallback import (
    ChunkFallbackStrategy,
)
from fitz_sage.engines.fitz_krag.retrieval.strategies.code_search import (
    CodeSearchStrategy,
)
from fitz_sage.engines.fitz_krag.retrieval.strategies.section_search import (
    SectionSearchStrategy,
)
from fitz_sage.engines.fitz_krag.retrieval.strategies.table_search import (
    TableSearchStrategy,
)


class TestRetrievalModeSchema:
    """The schema field is wired and defaults to hybrid."""

    def test_default_is_hybrid(self) -> None:
        cfg = FitzKragConfig(collection="test")
        assert cfg.retrieval_mode == "hybrid"

    def test_chat_only_is_accepted(self) -> None:
        cfg = FitzKragConfig(collection="test", retrieval_mode="chat_only")
        assert cfg.retrieval_mode == "chat_only"

    def test_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(Exception):
            FitzKragConfig(collection="test", retrieval_mode="weird")  # type: ignore[arg-type]


class TestSectionStrategyChatOnly:
    """SectionSearchStrategy with embedder=None skips semantic search."""

    def _make(self, embedder):
        store = MagicMock()
        store.search_bm25.return_value = [
            {"id": "s1", "raw_file_id": "f", "title": "T", "level": 1},
        ]
        store.search_by_keywords.return_value = []
        cfg = FitzKragConfig(collection="test")
        return SectionSearchStrategy(store, embedder, cfg), store

    def test_no_embedder_no_semantic_call(self) -> None:
        strategy, store = self._make(embedder=None)
        results = strategy.retrieve("query", limit=5)
        # BM25 results came through as Address objects.
        assert len(results) == 1
        # Semantic search must NOT have been called.
        store.search_by_vector.assert_not_called()

    def test_with_embedder_does_run_semantic(self) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 4
        strategy, store = self._make(embedder=embedder)
        store.search_by_vector.return_value = []
        strategy.retrieve("query", limit=5)
        embedder.embed.assert_called_once()
        store.search_by_vector.assert_called_once()


class TestCodeStrategyChatOnly:
    """CodeSearchStrategy with embedder=None skips semantic search."""

    def _make(self, embedder):
        store = MagicMock()
        store.search_by_name.return_value = [
            {"id": "c1", "raw_file_id": "f", "name": "foo", "qualified_name": "m.foo",
             "kind": "function", "start_line": 1, "end_line": 10},
        ]
        store.search_bm25.return_value = []
        store.search_by_keywords.return_value = []
        cfg = FitzKragConfig(collection="test")
        return CodeSearchStrategy(store, embedder, cfg), store

    def test_no_embedder_no_semantic_call(self) -> None:
        strategy, store = self._make(embedder=None)
        results = strategy.retrieve("query", limit=5)
        assert len(results) == 1
        store.search_by_vector.assert_not_called()

    def test_with_embedder_does_run_semantic(self) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 4
        strategy, store = self._make(embedder=embedder)
        store.search_by_vector.return_value = []
        strategy.retrieve("query", limit=5)
        embedder.embed.assert_called_once()
        store.search_by_vector.assert_called_once()


class TestTableStrategyChatOnly:
    """TableSearchStrategy with embedder=None skips semantic search."""

    def _make(self, embedder):
        store = MagicMock()
        store.search_by_name.return_value = []
        cfg = FitzKragConfig(collection="test")
        return TableSearchStrategy(store, embedder, cfg), store

    def test_no_embedder_no_semantic_call(self) -> None:
        strategy, store = self._make(embedder=None)
        results = strategy.retrieve("query", limit=5)
        assert results == []
        store.search_by_vector.assert_not_called()


class TestChunkFallbackChatOnly:
    """ChunkFallback is purely dense; chat-only mode short-circuits to []."""

    def test_no_embedder_returns_empty(self) -> None:
        vector_db = MagicMock()
        cfg = FitzKragConfig(collection="test")
        strategy = ChunkFallbackStrategy(vector_db, embedder=None, config=cfg)

        result = strategy.retrieve("query", limit=5)

        assert result == []
        vector_db.search.assert_not_called()


class TestEngineChatOnlyInit:
    """The engine init path doesn't construct an embedder in chat_only mode."""

    def test_chat_only_skips_embedder_factory(self, monkeypatch) -> None:
        """get_embedder must NOT be called when retrieval_mode == 'chat_only'."""
        from unittest.mock import patch

        # We only need to verify the high-level decision: the engine's
        # init pool should not submit an embedder future. We patch
        # everything heavy and inspect the LLM-factory call.

        get_embedder_calls: list = []

        def fake_get_embedder(*args, **kwargs):
            get_embedder_calls.append((args, kwargs))
            return MagicMock(dimensions=1024)

        with patch(
            "fitz_sage.llm.client.get_embedder", side_effect=fake_get_embedder
        ), patch("fitz_sage.llm.client.get_chat", return_value=MagicMock()), patch(
            "fitz_sage.storage.postgres.PostgresConnectionManager"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.schema.ensure_schema"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.import_graph_store.ImportGraphStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.raw_file_store.RawFileStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.section_store.SectionStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.symbol_store.SymbolStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.table_store.TableStore"
        ), patch(
            "fitz_sage.tabular.store.postgres.PostgresTableStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.router.RetrievalRouter"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.reader.ContentReader"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.expander.CodeExpander"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.code_search.CodeSearchStrategy"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.table_search.TableSearchStrategy"
        ), patch(
            "fitz_sage.engines.fitz_krag.query_analyzer.QueryAnalyzer"
        ), patch(
            "fitz_sage.engines.fitz_krag.context.assembler.ContextAssembler"
        ), patch(
            "fitz_sage.engines.fitz_krag.generation.synthesizer.CodeSynthesizer"
        ), patch(
            "fitz_sage.llm.factory.get_chat_factory", return_value=MagicMock()
        ):
            from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

            cfg = FitzKragConfig(collection="test", retrieval_mode="chat_only")
            engine = FitzKragEngine(cfg)

            # Embedder must be None.
            assert engine._embedder is None
            # And we never asked for an embedder.
            assert get_embedder_calls == []

    def test_hybrid_mode_constructs_embedder(self, monkeypatch) -> None:
        """The default 'hybrid' mode still builds an embedder (regression check)."""
        from unittest.mock import patch

        get_embedder_calls: list = []

        def fake_get_embedder(*args, **kwargs):
            get_embedder_calls.append((args, kwargs))
            embedder = MagicMock()
            embedder.dimensions = 1024
            return embedder

        with patch(
            "fitz_sage.llm.client.get_embedder", side_effect=fake_get_embedder
        ), patch("fitz_sage.llm.client.get_chat", return_value=MagicMock()), patch(
            "fitz_sage.storage.postgres.PostgresConnectionManager"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.schema.ensure_schema"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.import_graph_store.ImportGraphStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.raw_file_store.RawFileStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.section_store.SectionStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.symbol_store.SymbolStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.ingestion.table_store.TableStore"
        ), patch(
            "fitz_sage.tabular.store.postgres.PostgresTableStore"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.router.RetrievalRouter"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.reader.ContentReader"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.expander.CodeExpander"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.code_search.CodeSearchStrategy"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy"
        ), patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.table_search.TableSearchStrategy"
        ), patch(
            "fitz_sage.engines.fitz_krag.query_analyzer.QueryAnalyzer"
        ), patch(
            "fitz_sage.engines.fitz_krag.context.assembler.ContextAssembler"
        ), patch(
            "fitz_sage.engines.fitz_krag.generation.synthesizer.CodeSynthesizer"
        ), patch(
            "fitz_sage.llm.factory.get_chat_factory", return_value=MagicMock()
        ):
            from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

            cfg = FitzKragConfig(collection="test")  # default hybrid
            engine = FitzKragEngine(cfg)

            assert engine._embedder is not None
            assert len(get_embedder_calls) == 1
