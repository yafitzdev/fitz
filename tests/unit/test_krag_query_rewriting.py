# tests/unit/test_krag_query_rewriting.py
"""
Unit tests for query rewriting in FitzKragEngine.

Tests that the engine's _query_rewriter (when present) is called during the
answer() pipeline, and that failures or absence are handled gracefully.
"""

from __future__ import annotations

import pytest

from fitz_sage.core import Answer, Provenance, Query
from fitz_sage.core.exceptions import QueryIntelligenceError
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from tests.unit.mock_engine import build_mock_engine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Mock engine builder is shared across test_krag_{detection,engine,query_rewriting}.
_make_engine = build_mock_engine


def _make_query(text: str = "How does auth work?") -> Query:
    """Return a query with the given text."""
    return Query(text=text)


def _wire_happy_path(engine: FitzKragEngine, query_text: str) -> Answer:
    """Wire up all pipeline stages to return valid data for a full flow."""
    address = Address(
        kind=AddressKind.SECTION,
        source_id="auth-doc",
        location="Authentication",
        summary="Authentication implementation",
        score=0.9,
    )
    engine._retrieval_router.retrieve.return_value = [address]

    read_result = ReadResult(
        address=address,
        content="The authentication module validates login sessions.",
        file_path="auth.md",
    )
    engine._reader.read.return_value = [read_result]
    engine._expander.expand.return_value = [read_result]

    context = "[S1] Authentication implementation"
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
        engine = _make_engine(query_intelligence="endpoint/qwen2.5-7b-instruct")
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
        call_args = engine._retrieval_router.retrieve.call_args_list[0]
        assert call_args[0][0] == rewritten
        assert call_args[1]["rewrite_result"] is rewrite_result

        assert result is expected

    def test_original_query_used_when_rewrite_returns_same_text(self):
        """When batch_classify returns rewrite_result with same text, original flows through."""
        engine = _make_engine(query_intelligence="endpoint/qwen2.5-7b-instruct")
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
        call_args = engine._retrieval_router.retrieve.call_args_list[0]
        assert call_args[0][0] == query.text

        assert result is expected

    def test_query_intelligence_error_is_not_silently_downgraded(self):
        """When configured query intelligence fails, the query fails visibly."""
        engine = _make_engine(query_intelligence="endpoint/qwen2.5-7b-instruct")
        query = _make_query(
            "How does the authentication system work when handling multiple sessions?"
        )

        assert engine._query_rewriter is None
        engine._query_batcher.batch_classify.side_effect = QueryIntelligenceError("LLM timeout")

        _wire_happy_path(engine, query.text)

        with pytest.raises(QueryIntelligenceError, match="LLM timeout"):
            engine.answer(query)
        engine._query_batcher.batch_classify.assert_called_once()
        engine._retrieval_router.retrieve.assert_not_called()

    def test_rewriting_skipped_when_rewriter_is_none(self):
        """When _query_rewriter is None, the original query flows through directly."""
        engine = _make_engine()
        query = _make_query(
            "Where is the UserService class defined and what methods does it expose?"
        )
        assert engine._query_rewriter is None

        expected = _wire_happy_path(engine, query.text)

        result = engine.answer(query)

        # Router uses original query with no rewrite_result
        call_args = engine._retrieval_router.retrieve.call_args_list[0]
        assert call_args[0][0] == query.text
        assert call_args[1]["rewrite_result"] is None

        assert result is expected
