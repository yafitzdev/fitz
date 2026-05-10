# tests/unit/test_krag_multi_query.py
"""
Unit tests for multi-query expansion in RetrievalRouter.

Tests that long queries are expanded via LLM into sub-queries, short queries
skip expansion, and expansion failures are handled gracefully.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    top_addresses: int = 10,
    enable_multi_query: bool = True,
    multi_query_min_length: int = 300,
) -> MagicMock:
    """Create a mock FitzKragConfig with multi-query-relevant fields."""
    cfg = MagicMock()
    cfg.top_addresses = top_addresses
    cfg.enable_multi_query = enable_multi_query
    cfg.multi_query_min_length = multi_query_min_length
    return cfg


def _addr(
    location: str = "mod.func",
    score: float = 0.5,
    source_id: str = "src",
) -> Address:
    """Build an Address."""
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=f"Summary of {location}",
        score=score,
    )


def _long_query(length: int = 350) -> str:
    """Generate a query string of at least the given length."""
    base = "Explain in detail how the authentication module interacts with the session manager "
    repeats = (length // len(base)) + 1
    return (base * repeats)[:length]


# ---------------------------------------------------------------------------
# TestMultiQueryExpansion
# ---------------------------------------------------------------------------


class TestMultiQueryExpansion:
    """Tests for multi-query expansion in RetrievalRouter.retrieve()."""

    def test_long_query_triggers_expansion(self):
        """Query exceeding min_length triggers LLM-based expansion into sub-queries."""
        code_strat = MagicMock(name="code_strategy")
        code_strat.retrieve.return_value = [_addr(location="auth.login")]

        chat_factory = MagicMock(name="chat_factory")
        mock_chat = MagicMock(name="chat_client")
        # LLM returns JSON array of sub-queries
        mock_chat.chat.return_value = json.dumps(
            [
                "How does authentication work?",
                "How does session management work?",
            ]
        )
        chat_factory.return_value = mock_chat

        config = _make_config(
            enable_multi_query=True,
            multi_query_min_length=100,
        )

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            chat_factory=chat_factory,
        )

        long_q = _long_query(150)
        profile = RetrievalProfile(run_multi_query=True)
        router.retrieve(long_q, profile)

        # chat_factory called with "fast" to get cheap LLM
        chat_factory.assert_called_once_with("fast")

        # LLM chat called with prompt containing the query
        mock_chat.chat.assert_called_once()
        prompt_arg = mock_chat.chat.call_args[0][0]
        assert any(long_q in msg["content"] for msg in prompt_arg)

        # Code strategy called multiple times: original + 2 sub-queries = 3
        assert code_strat.retrieve.call_count == 3

    def test_short_query_skips_expansion(self):
        """Query shorter than min_length does NOT trigger expansion."""
        code_strat = MagicMock(name="code_strategy")
        code_strat.retrieve.return_value = [_addr()]

        chat_factory = MagicMock(name="chat_factory")

        config = _make_config(
            enable_multi_query=True,
            multi_query_min_length=300,
        )

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            chat_factory=chat_factory,
        )

        short_q = "How does auth work?"
        assert len(short_q) < 300

        router.retrieve(short_q)

        # chat_factory never called
        chat_factory.assert_not_called()

        # Code strategy called exactly once (original query only)
        code_strat.retrieve.assert_called_once()

    def test_expansion_disabled_skips_llm(self):
        """When enable_multi_query is False, no expansion regardless of length."""
        code_strat = MagicMock(name="code_strategy")
        code_strat.retrieve.return_value = [_addr()]

        chat_factory = MagicMock(name="chat_factory")

        config = _make_config(
            enable_multi_query=False,
            multi_query_min_length=100,
        )

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            chat_factory=chat_factory,
        )

        router.retrieve(_long_query(200))

        chat_factory.assert_not_called()
        code_strat.retrieve.assert_called_once()

    def test_expansion_skipped_when_no_chat_factory(self):
        """When _chat_factory is None, expansion does not run."""
        code_strat = MagicMock(name="code_strategy")
        code_strat.retrieve.return_value = [_addr()]

        config = _make_config(
            enable_multi_query=True,
            multi_query_min_length=100,
        )

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            chat_factory=None,
        )

        router.retrieve(_long_query(200))

        # Only one retrieval call (no expansion)
        code_strat.retrieve.assert_called_once()

    def test_expansion_failure_does_not_break_routing(self):
        """If LLM call fails during expansion, retrieval proceeds with original query."""
        code_strat = MagicMock(name="code_strategy")
        code_strat.retrieve.return_value = [_addr(location="auth.validate")]

        chat_factory = MagicMock(name="chat_factory")
        mock_chat = MagicMock(name="chat_client")
        mock_chat.chat.side_effect = RuntimeError("LLM rate limit exceeded")
        chat_factory.return_value = mock_chat

        config = _make_config(
            enable_multi_query=True,
            multi_query_min_length=100,
        )

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            chat_factory=chat_factory,
        )

        profile = RetrievalProfile(run_multi_query=True)
        result = router.retrieve(_long_query(200), profile)

        # chat_factory was called (expansion was attempted)
        chat_factory.assert_called_once()

        # Code strategy called once (original query only, expansion returned [])
        code_strat.retrieve.assert_called_once()

        # Results still returned
        assert len(result) == 1
        assert result[0].location == "auth.validate"

    def test_expansion_with_invalid_json_falls_back(self):
        """If LLM returns non-JSON, expansion returns empty and search proceeds."""
        code_strat = MagicMock(name="code_strategy")
        code_strat.retrieve.return_value = [_addr()]

        chat_factory = MagicMock(name="chat_factory")
        mock_chat = MagicMock(name="chat_client")
        mock_chat.chat.return_value = "This is not valid JSON at all"
        chat_factory.return_value = mock_chat

        config = _make_config(
            enable_multi_query=True,
            multi_query_min_length=100,
        )

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            chat_factory=chat_factory,
        )

        result = router.retrieve(_long_query(200))

        # Only original query searched
        code_strat.retrieve.assert_called_once()
        assert len(result) > 0
