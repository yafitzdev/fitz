# tests/unit/test_krag_query_rewriting.py
"""
Unit tests for query rewriting in FitzKragEngine.

Tests that the engine's _query_rewriter (when present) is called during the
answer() pipeline, and that failures or absence are handled gracefully.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.core import Answer, Provenance
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
from tests.unit.mock_engine import build_mock_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Mock engine builder is shared across test_krag_{detection,engine,query_rewriting}.
_make_engine = build_mock_engine


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
        """batch_classify returns a rewrite_result; rewritten query is used for retrieval."""
        engine = _make_engine()
        query = _make_query(
            "How does the authentication system handle user login sessions securely?"
        )

        from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
        from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
        from fitz_sage.retrieval.rewriter.types import RewriteResult, RewriteType

        rewritten = "authentication module implementation for secure user login session handling"
        rewrite_result = RewriteResult(
            original_query=query.text,
            rewritten_query=rewritten,
            rewrite_type=RewriteType.RETRIEVAL,
            confidence=0.9,
        )
        batch_result = BatchResult(
            analysis=QueryAnalysis(
                primary_type=QueryType.GENERAL, confidence=0.8, refined_query=rewritten
            ),
            rewrite_result=rewrite_result,
        )
        engine._query_batcher.batch_classify.side_effect = None
        engine._query_batcher.batch_classify.return_value = batch_result

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # batch_classify was called (once, with the sanitized original query)
        engine._query_batcher.batch_classify.assert_called_once()
        batch_call_args = engine._query_batcher.batch_classify.call_args
        assert batch_call_args[0][0] == query.text

        # Router receives the rewritten query and the rewrite_result
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0][0] == rewritten
        assert call_args[1]["rewrite_result"] is rewrite_result

        assert result is expected

    def test_original_query_used_when_rewrite_returns_same_text(self):
        """When batch_classify returns rewrite_result with same text, original flows through."""
        engine = _make_engine()
        query = _make_query("What is the login function and how does it validate user credentials?")

        from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
        from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
        from fitz_sage.retrieval.rewriter.types import RewriteResult, RewriteType

        rewrite_result = RewriteResult(
            original_query=query.text,
            rewritten_query=query.text,
            rewrite_type=RewriteType.NONE,
            confidence=0.0,
        )
        batch_result = BatchResult(
            analysis=QueryAnalysis(
                primary_type=QueryType.GENERAL, confidence=0.8, refined_query=query.text
            ),
            rewrite_result=rewrite_result,
        )
        engine._query_batcher.batch_classify.side_effect = None
        engine._query_batcher.batch_classify.return_value = batch_result

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

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
