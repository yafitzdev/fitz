"""Tests for batched query intelligence fallbacks."""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.query_batcher import QueryBatcher
from fitz_sage.engines.fitz_krag.query_analyzer import QueryType
from fitz_sage.retrieval.rewriter.types import RewriteType


def test_query_batcher_latches_fallback_after_provider_failure() -> None:
    """A failed local provider should fall back deterministically for later queries."""
    chat_factory = MagicMock(side_effect=RuntimeError("onnx provider unavailable"))
    batcher = QueryBatcher(chat_factory=chat_factory)

    first = batcher.batch_classify("What is the Acme refund window?")
    second = batcher.batch_classify("What is the incident acknowledgement target?")

    assert batcher.fallback_only is True
    assert chat_factory.call_count == 1
    assert first.analysis.primary_type is QueryType.GENERAL
    assert first.rewrite_result.rewrite_type is RewriteType.NONE
    assert second.analysis.primary_type is QueryType.GENERAL
    assert second.rewrite_result.rewrite_type is RewriteType.NONE
