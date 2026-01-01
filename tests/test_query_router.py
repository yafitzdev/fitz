# tests/test_query_router.py
"""Tests for the semantic query routing module."""

from __future__ import annotations

import pytest

from fitz_ai.engines.fitz_rag.routing import QueryIntent, QueryRouter


class MockEmbedder:
    """
    Mock embedder that returns predictable vectors.

    Global exemplars and global-like queries return [1, 0, 0]
    Local queries return [0, 1, 0]
    """

    GLOBAL_PHRASES = {
        # Exemplars from GLOBAL_QUERY_EXEMPLARS
        "give me an overview of this",
        "summarize the main topics",
        "what are the key themes",
        "what is the big picture",
        "what are the trends",
        "tell me about this project",
        "what is this about",
        "what are the main points",
        "provide a high-level summary",
        "what patterns do you see across all",
        "give me a general sense",
        "what are the overall findings",
        # Similar queries that should match
        "overview of the codebase",
        "summarize everything",
        "what are the main topics",
        "high level summary please",
    }

    def embed(self, text: str) -> list[float]:
        """Return predictable vectors based on text content."""
        text_lower = text.lower()

        # Check if text is similar to any global phrase
        for phrase in self.GLOBAL_PHRASES:
            if phrase in text_lower or text_lower in phrase:
                return [1.0, 0.0, 0.0]  # Global vector

        # Check for global keywords as fallback
        global_keywords = ["overview", "summary", "summarize", "trends", "big picture"]
        for keyword in global_keywords:
            if keyword in text_lower:
                return [0.9, 0.1, 0.0]  # Mostly global

        # Default to local vector
        return [0.0, 1.0, 0.0]


class TestQueryRouterSemantic:
    """Tests for semantic query classification."""

    def test_global_query_overview(self):
        """Queries asking for overview should be classified as GLOBAL."""
        embedder = MockEmbedder()
        router = QueryRouter(embedder=embedder.embed, threshold=0.7)

        assert router.classify("give me an overview of this") == QueryIntent.GLOBAL
        assert router.classify("overview of the codebase") == QueryIntent.GLOBAL

    def test_global_query_summary(self):
        """Queries asking for summary should be classified as GLOBAL."""
        embedder = MockEmbedder()
        router = QueryRouter(embedder=embedder.embed, threshold=0.7)

        assert router.classify("summarize the main topics") == QueryIntent.GLOBAL
        assert router.classify("summarize everything") == QueryIntent.GLOBAL

    def test_global_query_trends(self):
        """Queries about trends should be classified as GLOBAL."""
        embedder = MockEmbedder()
        router = QueryRouter(embedder=embedder.embed, threshold=0.7)

        assert router.classify("what are the trends") == QueryIntent.GLOBAL

    def test_local_query_specific(self):
        """Specific queries should be classified as LOCAL."""
        embedder = MockEmbedder()
        router = QueryRouter(embedder=embedder.embed, threshold=0.7)

        assert router.classify("What is the function signature of parse_config?") == QueryIntent.LOCAL
        assert router.classify("How do I install the package?") == QueryIntent.LOCAL
        assert router.classify("Show me the error handling code") == QueryIntent.LOCAL

    def test_disabled_router_returns_local(self):
        """When disabled, router should always return LOCAL."""
        embedder = MockEmbedder()
        router = QueryRouter(enabled=False, embedder=embedder.embed, threshold=0.7)

        assert router.classify("give me an overview") == QueryIntent.LOCAL
        assert router.classify("summarize everything") == QueryIntent.LOCAL

    def test_no_embedder_returns_local(self):
        """When no embedder is provided, router should return LOCAL."""
        router = QueryRouter(embedder=None, threshold=0.7)

        assert router.classify("give me an overview") == QueryIntent.LOCAL
        assert router.classify("summarize everything") == QueryIntent.LOCAL

    def test_threshold_affects_classification(self):
        """Higher threshold should be more strict."""
        embedder = MockEmbedder()

        # With lower threshold, borderline cases should be GLOBAL
        router_loose = QueryRouter(embedder=embedder.embed, threshold=0.5)
        assert router_loose.classify("trends") == QueryIntent.GLOBAL

        # With very high threshold, only exact matches should be GLOBAL
        router_strict = QueryRouter(embedder=embedder.embed, threshold=0.99)
        # Only exact exemplar matches return [1.0, 0.0, 0.0] which has similarity 1.0
        assert router_strict.classify("give me an overview of this") == QueryIntent.GLOBAL
        # Queries without global keywords return [0.0, 1.0, 0.0] which has similarity 0.0
        assert router_strict.classify("how do I install this package?") == QueryIntent.LOCAL


class TestQueryRouterFilter:
    """Tests for L2 filter generation."""

    def test_get_l2_filter(self):
        """L2 filter should target hierarchy summaries."""
        router = QueryRouter()

        filter_dict = router.get_l2_filter()

        assert "must" in filter_dict
        assert len(filter_dict["must"]) == 1
        assert filter_dict["must"][0]["key"] == "is_hierarchy_summary"
        assert filter_dict["must"][0]["match"]["value"] is True


class TestQueryRouterCaching:
    """Tests for exemplar vector caching."""

    def test_exemplar_vectors_cached(self):
        """Exemplar vectors should be computed once and cached."""
        call_count = 0

        def counting_embedder(text: str) -> list[float]:
            nonlocal call_count
            call_count += 1
            return [1.0, 0.0, 0.0]

        router = QueryRouter(embedder=counting_embedder, threshold=0.7)

        # First call should embed all exemplars + the query
        router.classify("test query 1")
        first_call_count = call_count

        # Second call should only embed the new query (exemplars cached)
        router.classify("test query 2")
        second_call_count = call_count - first_call_count

        # Should only have 1 new embedding call (just the query)
        assert second_call_count == 1
