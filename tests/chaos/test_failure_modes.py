"""Failure and recovery tests for the retrieval-first product surface."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from fitz_sage.core import Query
from fitz_sage.core.exceptions import KnowledgeError

pytestmark = pytest.mark.chaos


@pytest.fixture
def engine(krag_e2e_runner):
    return krag_e2e_runner.engine


def test_unknown_query_remains_bounded(engine) -> None:
    """An unknown query returns a normal bounded retrieval result."""
    results = engine.retrieve(Query(text="xyzzy12345 nonexistent query that matches nothing"))

    assert isinstance(results, list)
    assert len(results) <= engine.config.top_read


def test_retrieval_exception_is_wrapped(engine) -> None:
    """A strategy failure crosses the engine boundary as KnowledgeError."""
    with patch.object(
        engine._retrieval_router,
        "retrieve",
        side_effect=ConnectionError("SQLite read failed"),
    ):
        with pytest.raises(KnowledgeError, match="Retrieval failed"):
            engine.retrieve(Query(text="What is TechCorp?"))


def test_reranker_failure_uses_ranked_recall_fallback(engine) -> None:
    """The reranker owns a deterministic candidate-order fallback."""
    if not engine._address_reranker:
        pytest.skip("No reranker configured")

    with patch.object(
        engine._address_reranker._reranker,
        "rerank",
        side_effect=RuntimeError("Reranker unavailable"),
    ):
        results = engine.retrieve(Query(text="What is TechCorp?"))

    assert isinstance(results, list)


def test_memory_failure_is_not_silently_returned_as_success(engine) -> None:
    """Resource exhaustion must propagate through an explicit error boundary."""
    with patch.object(
        engine._retrieval_router,
        "retrieve",
        side_effect=MemoryError("Out of memory"),
    ):
        with pytest.raises((KnowledgeError, MemoryError)):
            engine.retrieve(Query(text="What is TechCorp?"))


def test_engine_recovers_after_transient_retrieval_error(engine) -> None:
    """A failed query must not poison the next query."""
    original_retrieve = engine._retrieval_router.retrieve
    call_count = 0

    def flaky_retrieve(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Transient failure")
        return original_retrieve(*args, **kwargs)

    with patch.object(engine._retrieval_router, "retrieve", side_effect=flaky_retrieve):
        with pytest.raises(KnowledgeError):
            engine.retrieve(Query(text="What is TechCorp?"))
        recovered = engine.retrieve(Query(text="What is TechCorp?"))

    assert isinstance(recovered, list)
