# tests/test_query_router.py
"""Tests for the query routing module."""

from __future__ import annotations

import pytest

from fitz_ai.engines.fitz_rag.routing import QueryIntent, QueryRouter


class TestQueryRouter:
    """Tests for QueryRouter classification."""

    def test_local_query_default(self):
        """Specific queries should be classified as LOCAL."""
        router = QueryRouter()

        local_queries = [
            "What is the function signature of parse_config?",
            "How do I install the package?",
            "Where is the database connection defined?",
            "Show me the error handling code",
            "What arguments does the CLI accept?",
        ]

        for query in local_queries:
            assert router.classify(query) == QueryIntent.LOCAL, f"Expected LOCAL for: {query}"

    def test_global_query_overview(self):
        """Queries with 'overview' pattern should be classified as GLOBAL."""
        router = QueryRouter()

        global_queries = [
            "Give me an overview of the codebase",
            "What is the overall architecture?",
            "Summarize the main components",
            "What are the key themes in this project?",
            "What are the main topics covered?",
            "What is this about?",
            "Tell me about this project",
        ]

        for query in global_queries:
            assert router.classify(query) == QueryIntent.GLOBAL, f"Expected GLOBAL for: {query}"

    def test_global_query_trends(self):
        """Queries about trends should be classified as GLOBAL."""
        router = QueryRouter()

        assert router.classify("What trends do you see?") == QueryIntent.GLOBAL
        assert router.classify("What are the trends in the data?") == QueryIntent.GLOBAL

    def test_global_query_summary(self):
        """Queries asking for summary should be classified as GLOBAL."""
        router = QueryRouter()

        assert router.classify("Summarize the project") == QueryIntent.GLOBAL
        assert router.classify("Give me a summary") == QueryIntent.GLOBAL

    def test_global_query_high_level(self):
        """High-level queries should be classified as GLOBAL."""
        router = QueryRouter()

        assert router.classify("What's the high-level architecture?") == QueryIntent.GLOBAL
        assert router.classify("Give me a high level view") == QueryIntent.GLOBAL

    def test_global_query_across_all(self):
        """Cross-corpus queries should be classified as GLOBAL."""
        router = QueryRouter()

        assert router.classify("What patterns appear across all files?") == QueryIntent.GLOBAL

    def test_disabled_router_returns_local(self):
        """When disabled, router should always return LOCAL."""
        router = QueryRouter(enabled=False)

        assert router.classify("Give me an overview") == QueryIntent.LOCAL
        assert router.classify("Summarize everything") == QueryIntent.LOCAL
        assert router.classify("What are the main topics?") == QueryIntent.LOCAL

    def test_custom_patterns(self):
        """Custom patterns should trigger GLOBAL classification."""
        router = QueryRouter(custom_patterns=["my-keyword", "project-meta"])

        assert router.classify("Tell me about my-keyword") == QueryIntent.GLOBAL
        assert router.classify("What is the project-meta?") == QueryIntent.GLOBAL
        # Default patterns should still work
        assert router.classify("Give me an overview") == QueryIntent.GLOBAL
        # Non-matching should be LOCAL
        assert router.classify("What is the function?") == QueryIntent.LOCAL

    def test_case_insensitive(self):
        """Pattern matching should be case insensitive."""
        router = QueryRouter()

        assert router.classify("GIVE ME AN OVERVIEW") == QueryIntent.GLOBAL
        assert router.classify("Give Me An Overview") == QueryIntent.GLOBAL
        assert router.classify("SUMMARIZE this") == QueryIntent.GLOBAL


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
