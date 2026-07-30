"""Tests for strict batched query intelligence handling."""

from __future__ import annotations

import json

import pytest

from fitz_sage.core.exceptions import QueryIntelligenceError
from fitz_sage.engines.fitz_krag.query_analyzer import QueryType
from fitz_sage.engines.fitz_krag.query_batcher import QueryBatcher
from fitz_sage.retrieval.rewriter.types import RewriteType


class _Chat:
    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.options = {}

    def chat(self, messages, **kwargs):
        self.options = kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _factory(response: str | Exception):
    return lambda _tier: _Chat(response)


def test_batch_classify_parses_requested_sections() -> None:
    response = json.dumps(
        {
            "analysis": {
                "primary_type": "documentation",
                "confidence": 0.88,
                "entities": ["refund"],
                "refined_query": "refund policy",
            },
            "rewriting": {
                "rewritten_query": "refund policy",
                "rewrite_type": "none",
                "confidence": 0.0,
                "is_compound": False,
                "decomposed_queries": [],
                "is_ambiguous": False,
                "disambiguated_queries": [],
            },
            "extended": {
                "specificity": "narrow",
                "answer_type": "factual",
                "domain": "general",
            },
            "keywords": ["refund", "returns"],
        }
    )
    batcher = QueryBatcher(chat_factory=_factory(response))

    result = batcher.batch_classify(
        "refund policy",
        include_analysis=True,
        include_detection=False,
        include_rewriting=True,
        include_extended=True,
        include_keywords=True,
    )

    assert result.analysis is not None
    assert result.analysis.primary_type is QueryType.DOCUMENTATION
    assert result.rewrite_result is not None
    assert result.rewrite_result.rewrite_type is RewriteType.NONE
    assert result.extended_signals == {
        "specificity": "narrow",
        "answer_type": "factual",
        "domain": "general",
    }
    assert "returns" in result.keywords


def test_batch_classify_propagates_provider_errors() -> None:
    batcher = QueryBatcher(chat_factory=_factory(RuntimeError("provider unavailable")))

    with pytest.raises(QueryIntelligenceError, match="provider unavailable"):
        batcher.batch_classify(
            "refund policy",
            include_analysis=False,
            include_detection=False,
            include_rewriting=False,
            include_extended=False,
            include_keywords=True,
        )


def test_batch_classify_rejects_missing_requested_sections() -> None:
    batcher = QueryBatcher(chat_factory=_factory("{}"))

    with pytest.raises(QueryIntelligenceError, match="missing `keywords` array"):
        batcher.batch_classify(
            "refund policy",
            include_analysis=False,
            include_detection=False,
            include_rewriting=False,
            include_extended=False,
            include_keywords=True,
        )


def test_batch_classify_filters_schema_placeholder_keywords() -> None:
    batcher = QueryBatcher(
        chat_factory=_factory(json.dumps({"keywords": ["term", "keyword", "return policy"]}))
    )

    result = batcher.batch_classify(
        "refund policy",
        include_analysis=False,
        include_detection=False,
        include_rewriting=False,
        include_extended=False,
        include_keywords=True,
    )

    assert "term" not in result.keywords
    assert "keyword" not in result.keywords
    assert "return policy" in result.keywords


def test_batch_classify_bounds_semantic_keyword_count() -> None:
    chat = _Chat(json.dumps({"keywords": [f"concept {index}" for index in range(15)]}))
    batcher = QueryBatcher(chat_factory=lambda _tier: chat)

    result = batcher.batch_classify(
        "refund policy",
        include_analysis=False,
        include_detection=False,
        include_rewriting=False,
        include_extended=False,
        include_keywords=True,
    )

    assert result.keywords == [f"concept {index}" for index in range(10)]


def test_batch_classify_applies_task_generation_budget() -> None:
    chat = _Chat(json.dumps({"keywords": ["return policy"]}))
    batcher = QueryBatcher(
        chat_factory=lambda _tier: chat,
        max_tokens=128,
    )

    batcher.batch_classify(
        "refund policy",
        include_analysis=False,
        include_detection=False,
        include_rewriting=False,
        include_extended=False,
        include_keywords=True,
    )

    assert chat.options == {"max_tokens": 128}


def test_keyword_prompt_requires_separate_json_array_items() -> None:
    batcher = QueryBatcher(chat_factory=lambda _tier: _Chat("{}"))

    prompt = batcher._build_prompt(
        "refund policy",
        include_analysis=False,
        include_detection=False,
        active_modules=[],
        include_rewriting=False,
        include_extended=False,
        include_keywords=True,
    )

    assert "every keyword as a separate quoted JSON array item" in prompt
    assert "never combine" in prompt
