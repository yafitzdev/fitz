# fitz_sage/llm/providers/llm_reranker.py
"""
LLM-based reranker — uses a chat model to score document relevance.

This is the canonical rerank backend in fitz-sage after the deletion
of the dedicated cohere/rerank provider. It implements the
``RerankProvider`` protocol so existing consumers
(``AddressReranker`` etc.) work without modification.

The reranker batches all candidates into a single chat call and asks
the model to emit a single integer relevance score per document, in
``<index>: <score>`` form. Scores are parsed defensively — any line
the model emits that doesn't match is ignored, and any documents the
model omits keep their original ordinal as a tiebreaker. This makes
the reranker robust to verbose models that prepend reasoning.

Tier choice: rerank is a structured-classification task and benefits
from a fast, cheap model. The default tier is ``"fast"``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Iterable

from fitz_sage.llm.providers.base import RerankResult

if TYPE_CHECKING:
    from fitz_sage.llm.factory import ChatFactory, ModelTier

logger = logging.getLogger(__name__)

# Matches "<doc-index>: <score>" with optional whitespace and decimals.
_SCORE_LINE = re.compile(r"^\s*(\d+)\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)\s*$", re.MULTILINE)

_RERANK_PROMPT = """\
You are an information-retrieval relevance grader.

Score each document below for its relevance to the query on a scale of
0 (completely irrelevant) to 10 (perfectly answers the query).

Output ONE LINE PER DOCUMENT in the exact form:

  <document-number>: <score>

Do not include any other text. Do not omit any document.

Query: {query}

Documents:
{documents}

Scores (one per line, "<n>: <0-10>"):"""


class LLMReranker:
    """Rerank candidates by asking a chat model to score each one.

    Args:
        chat_factory: A ``ChatFactory`` (from ``fitz_sage.llm.factory``)
            that produces chat clients per tier.
        tier: Which tier to use for reranking. Defaults to ``"fast"``
            since rerank is a cheap structured-classification task.
        max_doc_chars: Per-document truncation cap (characters) before
            the prompt is built. Keeps the rerank call cheap on long
            documents.
    """

    def __init__(
        self,
        chat_factory: "ChatFactory",
        tier: "ModelTier" = "fast",
        max_doc_chars: int = 1000,
    ) -> None:
        self._chat_factory = chat_factory
        self._tier: "ModelTier" = tier
        self._max_doc_chars = max_doc_chars

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Score and order ``documents`` by relevance to ``query``.

        The returned list is sorted by score descending and truncated
        to ``top_n`` if provided. Documents the model failed to score
        get a default score of 0 and stable ordinal as a tiebreaker.
        """
        if not documents:
            return []

        if len(documents) == 1:
            # Trivial case — no point burning a chat call.
            return [RerankResult(index=0, score=1.0)]

        chat = self._chat_factory(self._tier)
        prompt = self._build_prompt(query, documents)

        try:
            response = chat.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max(64, len(documents) * 8),
            )
        except Exception as e:
            logger.warning("LLMReranker chat call failed: %s", e)
            # Fall back to identity ordering.
            return [
                RerankResult(index=i, score=0.0)
                for i in range(len(documents) if top_n is None else min(top_n, len(documents)))
            ]

        scores = self._parse_scores(response, len(documents))

        results = [RerankResult(index=i, score=float(s)) for i, s in enumerate(scores)]
        results.sort(key=lambda r: (-r.score, r.index))

        if top_n is not None:
            results = results[:top_n]
        return results

    def _build_prompt(self, query: str, documents: list[str]) -> str:
        formatted_docs: list[str] = []
        for i, doc in enumerate(documents, start=1):
            truncated = doc if len(doc) <= self._max_doc_chars else doc[: self._max_doc_chars]
            formatted_docs.append(f"{i}. {truncated}")
        return _RERANK_PROMPT.format(query=query, documents="\n\n".join(formatted_docs))

    @staticmethod
    def _parse_scores(response: str, n_documents: int) -> list[float]:
        """Extract per-document scores from the model's response.

        Models occasionally prepend reasoning or use slightly different
        delimiters. We pull every ``<int>: <number>`` we can find and
        clamp it to [0, 10]. Missing documents default to 0.
        """
        scores = [0.0] * n_documents
        for match in _SCORE_LINE.finditer(response):
            idx = int(match.group(1)) - 1  # 1-indexed in the prompt
            try:
                value = float(match.group(2))
            except ValueError:
                continue
            if 0 <= idx < n_documents:
                scores[idx] = max(0.0, min(10.0, value))
        return scores

    @classmethod
    def from_iterable(
        cls,
        chat_factory: "ChatFactory",
        documents: Iterable[str],
        query: str,
        tier: "ModelTier" = "fast",
        top_n: int | None = None,
    ) -> list[RerankResult]:
        """Convenience: instantiate + rerank in one call (for tests / one-shots)."""
        return cls(chat_factory=chat_factory, tier=tier).rerank(query, list(documents), top_n=top_n)


__all__ = ["LLMReranker"]
