# tests/unit/llm/test_llm_reranker.py
"""
Unit tests for LLMReranker — the canonical rerank backend after the
cohere/rerank deletion.

The reranker batches candidates into a single chat call and parses
``<index>: <score>`` lines back from the response. Tests exercise:
- happy-path scoring + ordering
- top_n truncation
- robustness to messy responses (extra text, missing scores,
  out-of-range scores)
- single-document and empty-input edge cases
- chat failure -> identity fallback
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.llm.providers.base import RerankProvider, RerankResult
from fitz_sage.llm.providers.llm_reranker import LLMReranker


def _factory(response: str) -> tuple[MagicMock, MagicMock]:
    """Build a chat factory that returns a chat client emitting ``response``."""
    chat = MagicMock()
    chat.chat.return_value = response
    factory = MagicMock(return_value=chat)
    return factory, chat


class TestLLMRerankerProtocol:
    """LLMReranker satisfies the RerankProvider runtime protocol."""

    def test_implements_protocol(self) -> None:
        factory, _ = _factory("1: 1\n")
        rerank = LLMReranker(chat_factory=factory)
        assert isinstance(rerank, RerankProvider)


class TestLLMRerankerScoring:
    """Happy-path scoring and ordering."""

    def test_orders_by_score_desc(self) -> None:
        factory, chat = _factory("1: 3\n2: 9\n3: 5\n")
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("query", ["a", "b", "c"])
        assert [r.index for r in result] == [1, 2, 0]
        assert [r.score for r in result] == [9.0, 5.0, 3.0]
        chat.chat.assert_called_once()

    def test_uses_fast_tier_by_default(self) -> None:
        factory, _ = _factory("1: 8\n")
        rerank = LLMReranker(chat_factory=factory)
        rerank.rerank("q", ["only"])
        # Single doc short-circuit -> chat NOT called. Force a multi-doc
        # call to confirm tier.
        rerank.rerank("q", ["a", "b"])
        factory.assert_called_with("fast")

    def test_explicit_tier(self) -> None:
        factory, _ = _factory("1: 1\n2: 2\n")
        rerank = LLMReranker(chat_factory=factory, tier="balanced")
        rerank.rerank("q", ["a", "b"])
        factory.assert_called_with("balanced")

    def test_top_n_truncates(self) -> None:
        factory, _ = _factory("1: 3\n2: 9\n3: 5\n")
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b", "c"], top_n=2)
        assert len(result) == 2
        assert [r.index for r in result] == [1, 2]


class TestLLMRerankerEdgeCases:
    """Edge cases — empty/single inputs, malformed responses, failures."""

    def test_empty_documents(self) -> None:
        factory, _ = _factory("")
        rerank = LLMReranker(chat_factory=factory)
        assert rerank.rerank("q", []) == []
        # Chat must not be called for empty input.
        factory.assert_not_called()

    def test_single_document_short_circuits(self) -> None:
        factory, chat = _factory("")
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["only"])
        assert result == [RerankResult(index=0, score=1.0)]
        chat.chat.assert_not_called()
        factory.assert_not_called()

    def test_robust_to_verbose_response(self) -> None:
        """Models often prepend reasoning. Parser must extract regardless."""
        response = (
            "Sure, here are the scores after considering each document carefully:\n"
            "\n"
            "1: 8\n"
            "2: 2\n"
            "3: 6\n"
            "\n"
            "Hope that helps!"
        )
        factory, _ = _factory(response)
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b", "c"])
        assert [r.index for r in result] == [0, 2, 1]
        assert [r.score for r in result] == [8.0, 6.0, 2.0]

    def test_missing_score_defaults_to_zero(self) -> None:
        """A document the model forgot keeps a stable score of 0."""
        factory, _ = _factory("1: 7\n3: 4\n")  # 2 missing
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b", "c"])
        # Doc 0 -> 7, Doc 2 -> 4, Doc 1 -> 0
        assert result[0].index == 0 and result[0].score == 7.0
        assert result[1].index == 2 and result[1].score == 4.0
        assert result[2].index == 1 and result[2].score == 0.0

    def test_clamps_out_of_range_scores(self) -> None:
        factory, _ = _factory("1: -3\n2: 99\n")
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b"])
        # Both clamped: -3 -> 0, 99 -> 10
        assert result[0].score == 10.0
        assert result[1].score == 0.0

    def test_decimal_scores_accepted(self) -> None:
        factory, _ = _factory("1: 3.5\n2: 7.25\n")
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b"])
        assert {r.score for r in result} == {3.5, 7.25}

    def test_dash_separator_accepted(self) -> None:
        """Some models emit '1 - 5' instead of '1: 5'."""
        factory, _ = _factory("1 - 5\n2 - 8\n")
        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b"])
        assert [r.index for r in result] == [1, 0]

    def test_chat_failure_returns_identity_order(self) -> None:
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("api down")
        factory = MagicMock(return_value=chat)

        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b", "c"])

        assert [r.index for r in result] == [0, 1, 2]
        assert all(r.score == 0.0 for r in result)

    def test_chat_failure_with_top_n(self) -> None:
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("api down")
        factory = MagicMock(return_value=chat)

        rerank = LLMReranker(chat_factory=factory)
        result = rerank.rerank("q", ["a", "b", "c", "d"], top_n=2)

        assert len(result) == 2
        assert [r.index for r in result] == [0, 1]


class TestLLMRerankerPromptBuilding:
    """Prompt formatting + truncation."""

    def test_truncates_long_documents(self) -> None:
        factory, chat = _factory("1: 5\n2: 5\n")
        rerank = LLMReranker(chat_factory=factory, max_doc_chars=20)
        long_doc = "x" * 200
        rerank.rerank("q", [long_doc, "short"])
        prompt = chat.chat.call_args[0][0][0]["content"]
        # Long doc was truncated to 20 chars in the prompt
        assert "x" * 200 not in prompt
        assert "x" * 20 in prompt

    def test_query_appears_in_prompt(self) -> None:
        factory, chat = _factory("1: 5\n2: 5\n")
        rerank = LLMReranker(chat_factory=factory)
        rerank.rerank("the meaning of life", ["a", "b"])
        prompt = chat.chat.call_args[0][0][0]["content"]
        assert "the meaning of life" in prompt
