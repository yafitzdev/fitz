# tests/unit/test_krag_query_rewriting.py
"""
Unit tests for query rewriting in FitzKragEngine.

Tests that the engine's _query_rewriter (when present) is called during the
answer() pipeline, and that failures or absence are handled gracefully.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.core import Answer, Provenance
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> FitzKragConfig:
    """Create a minimal FitzKragConfig for testing."""
    defaults = {"collection": "test_collection"}
    defaults.update(overrides)
    return FitzKragConfig(**defaults)


def _make_engine(**config_overrides) -> FitzKragEngine:
    """
    Build a FitzKragEngine with every component replaced by a MagicMock.

    Bypasses __init__ entirely so no real imports are triggered.
    """
    config = _make_config(**config_overrides)
    engine = FitzKragEngine.__new__(FitzKragEngine)
    engine._config = config
    engine._chat = MagicMock(name="chat")
    engine._connection_manager = MagicMock(name="connection_manager")
    engine._raw_store = MagicMock(name="raw_store")
    engine._symbol_store = MagicMock(name="symbol_store")
    engine._import_store = MagicMock(name="import_store")
    engine._section_store = MagicMock(name="section_store")
    engine._query_analyzer = MagicMock(name="query_analyzer")
    engine._retrieval_router = MagicMock(name="retrieval_router")
    engine._reader = MagicMock(name="reader")
    engine._expander = MagicMock(name="expander")
    engine._table_handler = MagicMock(name="table_handler")
    engine._table_handler.process.side_effect = lambda q, results: results
    engine._assembler = MagicMock(name="assembler")
    engine._synthesizer = MagicMock(name="synthesizer")
    engine._governance = None
    engine._detection_orchestrator = None
    engine._query_rewriter = None
    engine._address_reranker = None
    engine._hop_controller = None
    from fitz_sage.engines.fitz_krag.retrieval.retrieval_pass import RetrievalPass

    engine._retrieval_pass = RetrievalPass(
        engine._retrieval_router, engine._address_reranker, engine._reader, engine._config
    )
    engine._table_store = MagicMock(name="table_store")
    engine._pg_table_store = MagicMock(name="pg_table_store")
    engine._chat_factory = None
    engine._vocabulary_store = None
    engine._keyword_matcher = None
    engine._entity_graph_store = None
    engine._bg_worker = None
    engine._manifest = None
    engine._source_dir = None

    # Configure batcher to return sensible defaults so batched dispatch works
    from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
    from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
    from fitz_sage.retrieval.rewriter.types import RewriteResult, RewriteType

    def _default_batch_classify(query, **kwargs):
        return BatchResult(
            analysis=QueryAnalysis(
                primary_type=QueryType.GENERAL, confidence=0.8, refined_query=query
            ),
            detection_results=None,
            rewrite_result=RewriteResult(
                original_query=query,
                rewritten_query=query,
                rewrite_type=RewriteType.NONE,
                confidence=0.0,
            ),
        )

    engine._query_batcher = MagicMock(name="query_batcher")
    engine._query_batcher.batch_classify.side_effect = _default_batch_classify
    return engine


def _make_query(text: str = "How does auth work?") -> MagicMock:
    """Return a mock Query with the given text."""
    q = MagicMock(name="query")
    q.text = text
    return q


def _wire_happy_path(engine: FitzKragEngine, query_text: str) -> Answer:
    """Wire up all pipeline stages to return valid data for a full flow."""
    analysis = MagicMock(name="analysis")
    analysis.confidence = 0.8
    engine._query_analyzer.analyze.return_value = analysis

    address = MagicMock(name="addr")
    engine._retrieval_router.retrieve.return_value = [address]

    read_result = MagicMock(name="read")
    engine._reader.read.return_value = [read_result]
    engine._expander.expand.return_value = [read_result]

    context = MagicMock(name="context")
    engine._assembler.assemble.return_value = context

    expected = Answer(
        text="Answer text.",
        provenance=[Provenance(source_id="file.py:10")],
        metadata={"engine": "fitz_krag"},
    )
    engine._synthesizer.generate.return_value = expected
    return expected


# ---------------------------------------------------------------------------
# TestQueryRewriting
# ---------------------------------------------------------------------------


class TestQueryRewriting:
    """Tests for query rewriting integration in the answer() pipeline."""

    def test_rewrite_called_and_rewritten_query_used(self):
        """Rewriter called directly first; rewritten query used for retrieval and batcher."""
        engine = _make_engine()
        query = _make_query(
            "How does the authentication system handle user login sessions securely?"
        )

        # Enable rewriter and configure it to return a real RewriteResult
        from fitz_sage.retrieval.rewriter.types import RewriteResult, RewriteType

        rewritten = "authentication module implementation for secure user login session handling"
        rewrite_result = RewriteResult(
            original_query=query.text,
            rewritten_query=rewritten,
            rewrite_type=RewriteType.RETRIEVAL,
            confidence=0.9,
        )
        engine._query_rewriter = MagicMock(name="rewriter")
        engine._query_rewriter.rewrite.return_value = rewrite_result

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # Rewriter called directly with the sanitized query (Step 1)
        engine._query_rewriter.rewrite.assert_called_once_with(query.text)

        # Batcher called with the rewritten query for analysis (rewritten is 9 words > 8)
        engine._query_batcher.batch_classify.assert_called_once()
        batch_call_args = engine._query_batcher.batch_classify.call_args
        assert batch_call_args[0][0] == rewritten
        assert batch_call_args[1].get("include_rewriting") is False

        # Router receives the rewritten query and the rewrite_result
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0][0] == rewritten
        assert call_args[1]["rewrite_result"] is rewrite_result

        assert result is expected

    def test_original_query_used_when_rewrite_returns_same_text(self):
        """When rewriter returns same text, original query flows through unchanged."""
        engine = _make_engine()
        query = _make_query("What is the login function and how does it validate user credentials?")

        # Rewriter returns the original text unchanged
        from fitz_sage.retrieval.rewriter.types import RewriteResult, RewriteType

        rewrite_result = RewriteResult(
            original_query=query.text,
            rewritten_query=query.text,
            rewrite_type=RewriteType.NONE,
            confidence=0.0,
        )
        engine._query_rewriter = MagicMock(name="rewriter")
        engine._query_rewriter.rewrite.return_value = rewrite_result

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # Rewriter was called
        engine._query_rewriter.rewrite.assert_called_once_with(query.text)

        # Router uses original query (rewrite returned same text)
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0][0] == query.text

        assert result is expected

    def test_fallback_to_original_on_batch_error(self):
        """When batcher raises, the original query is used with fallback analysis."""
        engine = _make_engine()
        query = _make_query(
            "How does the authentication system work when handling multiple sessions?"
        )

        # No rewriter: rewrite_result stays None so fallback is clean
        assert engine._query_rewriter is None
        engine._query_batcher.batch_classify.side_effect = RuntimeError("LLM timeout")

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # Batcher was called (query is 10 words > 8, so LLM analysis needed) and failed
        engine._query_batcher.batch_classify.assert_called_once()

        # Falls back to original query text with no rewrite_result
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0][0] == query.text
        assert call_args[1]["rewrite_result"] is None

        assert result is expected

    def test_rewriting_skipped_when_rewriter_is_none(self):
        """When _query_rewriter is None, the original query flows through directly."""
        engine = _make_engine()
        query = _make_query(
            "Where is the UserService class defined and what methods does it expose?"
        )
        assert engine._query_rewriter is None

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # query_analyzer is not called in the new batched dispatch path
        engine._query_analyzer.analyze.assert_not_called()

        # Router uses original query with no rewrite_result
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0][0] == query.text
        assert call_args[1]["rewrite_result"] is None

        assert result is expected
