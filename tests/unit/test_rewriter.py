# tests/unit/test_rewriter.py
"""
Tests for fitz_sage.retrieval.rewriter module.

Tests cover:
1. ConversationContext - conversation history handling
2. RewriteResult - rewrite result handling
3. parse_rewrite_dict - JSON dict -> RewriteResult parsing
"""

from __future__ import annotations

from fitz_sage.retrieval.rewriter import (
    ConversationContext,
    ConversationMessage,
    RewriteResult,
    RewriteType,
    parse_rewrite_dict,
)

# ---------------------------------------------------------------------------
# Tests for ConversationContext
# ---------------------------------------------------------------------------


class TestConversationContext:
    """Tests for ConversationContext class."""

    def test_empty_context(self):
        """Test empty conversation context."""
        context = ConversationContext()
        assert context.is_empty()
        assert context.format_for_prompt() == ""

    def test_context_with_history(self):
        """Test context with conversation history."""
        context = ConversationContext(
            history=[
                ConversationMessage(role="user", content="Hello"),
                ConversationMessage(role="assistant", content="Hi there!"),
            ]
        )
        assert not context.is_empty()
        formatted = context.format_for_prompt()
        assert "User: Hello" in formatted
        assert "Assistant: Hi there!" in formatted

    def test_recent_history_limit(self):
        """Test that recent history is limited by max_turns."""
        messages = []
        for i in range(20):
            messages.append(ConversationMessage(role="user", content=f"Message {i}"))
            messages.append(ConversationMessage(role="assistant", content=f"Reply {i}"))

        context = ConversationContext(history=messages, max_turns=3)
        recent = context.recent_history()

        # Should only have last 6 messages (3 turns * 2 messages)
        assert len(recent) == 6

    def test_format_truncates_long_messages(self):
        """Test that long messages are truncated in formatting."""
        long_content = "A" * 600  # Longer than 500 char limit
        context = ConversationContext(
            history=[ConversationMessage(role="user", content=long_content)]
        )
        formatted = context.format_for_prompt()
        assert "..." in formatted
        assert len(formatted) < len(long_content)


# ---------------------------------------------------------------------------
# Tests for RewriteResult
# ---------------------------------------------------------------------------


class TestRewriteResult:
    """Tests for RewriteResult class."""

    def test_was_rewritten_none(self):
        """Test was_rewritten returns False for NONE type."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="test",
            rewrite_type=RewriteType.NONE,
            confidence=1.0,
        )
        assert not result.was_rewritten

    def test_was_rewritten_same_query(self):
        """Test was_rewritten returns False when queries are identical."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="test",
            rewrite_type=RewriteType.CLARITY,
            confidence=1.0,
        )
        assert not result.was_rewritten

    def test_was_rewritten_different_query(self):
        """Test was_rewritten returns True when query changed."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="improved test",
            rewrite_type=RewriteType.CLARITY,
            confidence=1.0,
        )
        assert result.was_rewritten

    def test_all_query_variations_no_rewrite(self):
        """Test query variations with no rewrite."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="test",
            rewrite_type=RewriteType.NONE,
            confidence=1.0,
        )
        variations = result.all_query_variations
        assert variations == ["test"]

    def test_all_query_variations_with_rewrite(self):
        """Test query variations with rewrite."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="improved test",
            rewrite_type=RewriteType.CLARITY,
            confidence=1.0,
        )
        variations = result.all_query_variations
        assert "test" in variations
        assert "improved test" in variations
        assert len(variations) == 2

    def test_all_query_variations_with_disambiguated(self):
        """Test query variations with disambiguated queries."""
        result = RewriteResult(
            original_query="test",
            rewritten_query="improved test",
            rewrite_type=RewriteType.COMBINED,
            confidence=0.8,
            is_ambiguous=True,
            disambiguated_queries=["interpretation 1", "interpretation 2"],
        )
        variations = result.all_query_variations
        assert len(variations) == 4
        assert "test" in variations
        assert "improved test" in variations
        assert "interpretation 1" in variations
        assert "interpretation 2" in variations


# ---------------------------------------------------------------------------
# Tests for parse_rewrite_dict
# ---------------------------------------------------------------------------


class TestParseRewriteDict:
    """Tests for parse_rewrite_dict function."""

    def test_parse_basic_clarity_rewrite(self):
        """Parse a basic clarity rewrite dict."""
        data = {
            "rewritten_query": "authentication module overview",
            "rewrite_type": "clarity",
            "confidence": 0.9,
            "is_ambiguous": False,
            "disambiguated_queries": [],
        }
        result = parse_rewrite_dict(data, "auth module overview")
        assert result.rewritten_query == "authentication module overview"
        assert result.rewrite_type == RewriteType.CLARITY
        assert result.confidence == 0.9
        assert result.original_query == "auth module overview"

    def test_parse_none_rewrite_type(self):
        """parse_rewrite_dict with rewrite_type=none produces NONE."""
        data = {
            "rewritten_query": "original query",
            "rewrite_type": "none",
            "confidence": 1.0,
            "is_ambiguous": False,
            "disambiguated_queries": [],
        }
        result = parse_rewrite_dict(data, "original query")
        assert result.rewrite_type == RewriteType.NONE
        assert not result.was_rewritten

    def test_parse_retrieval_rewrite_type(self):
        """parse_rewrite_dict with rewrite_type=retrieval produces RETRIEVAL."""
        data = {
            "rewritten_query": "improved retrieval query",
            "rewrite_type": "retrieval",
            "confidence": 0.85,
            "is_ambiguous": False,
            "disambiguated_queries": [],
        }
        result = parse_rewrite_dict(data, "original query")
        assert result.rewrite_type == RewriteType.RETRIEVAL

    def test_parse_decomposition_with_decomposed_queries(self):
        """Decomposition rewrite includes decomposed_queries (up to 5)."""
        data = {
            "rewritten_query": "compound query",
            "rewrite_type": "decomposition",
            "confidence": 0.8,
            "is_compound": True,
            "decomposed_queries": ["part one", "part two", "part three"],
            "is_ambiguous": False,
            "disambiguated_queries": [],
        }
        result = parse_rewrite_dict(data, "compound query")
        assert result.rewrite_type == RewriteType.DECOMPOSITION
        assert result.is_compound is True
        assert "part one" in result.decomposed_queries

    def test_parse_empty_rewritten_query_falls_back_to_original(self):
        """Empty rewritten_query falls back to original."""
        data = {
            "rewritten_query": "",
            "rewrite_type": "clarity",
            "confidence": 0.5,
            "is_ambiguous": False,
            "disambiguated_queries": [],
        }
        result = parse_rewrite_dict(data, "original query text")
        assert result.rewritten_query == "original query text"
        assert result.rewrite_type == RewriteType.NONE

    def test_parse_missing_fields_use_defaults(self):
        """Missing optional fields use sensible defaults."""
        data = {"rewritten_query": "some query"}
        result = parse_rewrite_dict(data, "original")
        assert result.rewritten_query == "some query"
        assert result.rewrite_type == RewriteType.NONE
        assert result.confidence == 0.5
        assert result.is_ambiguous is False
        assert result.disambiguated_queries == []

    def test_parse_disambiguated_queries_limited_to_three(self):
        """disambiguated_queries are capped at 3."""
        data = {
            "rewritten_query": "query",
            "rewrite_type": "combined",
            "confidence": 0.7,
            "is_ambiguous": True,
            "disambiguated_queries": ["q1", "q2", "q3", "q4", "q5"],
        }
        result = parse_rewrite_dict(data, "original")
        assert len(result.disambiguated_queries) == 3

    def test_parse_unknown_rewrite_type_falls_back_to_none(self):
        """Unknown rewrite_type string falls back to NONE."""
        data = {
            "rewritten_query": "some query",
            "rewrite_type": "invented_type_xyz",
            "confidence": 0.5,
        }
        result = parse_rewrite_dict(data, "original")
        assert result.rewrite_type == RewriteType.NONE

    def test_parse_conversational_type(self):
        """parse_rewrite_dict handles conversational rewrite type."""
        data = {
            "rewritten_query": "What are TechCorp's products?",
            "rewrite_type": "conversational",
            "confidence": 0.95,
            "is_ambiguous": False,
            "disambiguated_queries": [],
        }
        result = parse_rewrite_dict(data, "What are their products?")
        assert result.rewrite_type == RewriteType.CONVERSATIONAL
        assert result.rewritten_query == "What are TechCorp's products?"
        assert result.was_rewritten

    def test_parse_combined_type(self):
        """parse_rewrite_dict handles combined rewrite type."""
        data = {
            "rewritten_query": "combined improved query",
            "rewrite_type": "combined",
            "confidence": 0.75,
            "is_ambiguous": False,
            "disambiguated_queries": [],
        }
        result = parse_rewrite_dict(data, "original")
        assert result.rewrite_type == RewriteType.COMBINED
