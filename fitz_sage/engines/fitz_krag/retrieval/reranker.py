# fitz_sage/engines/fitz_krag/retrieval/reranker.py
"""
Address reranker for KRAG — cross-encoder reranking on address summaries.

Reranks retrieved addresses using a cross-encoder model before reading
their full content, improving precision of the top-k results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fitz_sage.engines.fitz_krag.retrieval.snippets import query_relevant_excerpt
from fitz_sage.engines.fitz_krag.retrieval.trace import addresses_trace
from fitz_sage.engines.fitz_krag.types import Address

if TYPE_CHECKING:
    from fitz_sage.llm.providers.base import RerankProvider

logger = logging.getLogger(__name__)

_MAX_RERANK_TEXT_CHARS = 1200


class AddressReranker:
    """Cross-encoder reranker for KRAG Address objects."""

    def __init__(
        self,
        reranker: "RerankProvider",
        k: int = 10,
        min_addresses: int = 2,
    ):
        self._reranker = reranker
        self._k = k
        self._min_addresses = min_addresses
        self.last_trace: dict[str, object] = {}

    def rerank(self, query: str, addresses: list[Address]) -> list[Address]:
        """
        Rerank addresses using cross-encoder on summaries.

        Skips reranking only when fewer than min_addresses are available.

        Args:
            query: User query text
            addresses: Retrieved addresses to rerank

        Returns:
            Reranked (and possibly truncated) list of addresses
        """
        if len(addresses) < self._min_addresses:
            selected = addresses[: self._k]
            self.last_trace = {
                "used": False,
                "reason": "below_min_addresses",
                "query": query,
                "input_count": len(addresses),
                "output_count": len(selected),
                "output": addresses_trace(selected),
            }
            return addresses[: self._k]

        documents = [_rerank_document(query, addr) for addr in addresses]

        try:
            ranked = self._reranker.rerank(query, documents, top_n=self._k)

            reranked: list[Address] = []
            ranked_trace = []
            for result in ranked:
                original = addresses[result.index]
                reranked_address = Address(
                    kind=original.kind,
                    source_id=original.source_id,
                    location=original.location,
                    summary=original.summary,
                    score=result.score,
                    metadata=original.metadata,
                )
                reranked.append(reranked_address)
                ranked_trace.append(
                    {
                        "original_index": result.index,
                        "score": result.score,
                        "address": addresses_trace([reranked_address])[0],
                    }
                )

            logger.debug(f"Reranked {len(addresses)} addresses to top {len(reranked)}")
            self.last_trace = {
                "used": True,
                "query": query,
                "input_count": len(addresses),
                "document_count": len(documents),
                "document_characters": [len(document) for document in documents],
                "provider": dict(getattr(self._reranker, "last_trace", {}) or {}),
                "output_count": len(reranked),
                "output": addresses_trace(reranked),
                "ranked": ranked_trace,
            }
            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            selected = addresses[: self._k]
            self.last_trace = {
                "used": False,
                "reason": "rerank_error",
                "error": str(e),
                "query": query,
                "input_count": len(addresses),
                "output_count": len(selected),
                "output": addresses_trace(selected),
            }
            return selected


def _rerank_document(query: str, addr: Address) -> str:
    """Build the text shown to the cross-encoder for one address."""
    rerank_heading = addr.metadata.get("rerank_heading")
    lead = (
        rerank_heading
        if isinstance(rerank_heading, str) and rerank_heading.strip()
        else addr.summary or addr.location
    )
    parts = [lead]
    text = addr.metadata.get("rerank_text")
    if not isinstance(text, str) or not text.strip():
        text = addr.metadata.get("text")
    if isinstance(text, str) and text.strip():
        parts.append(
            query_relevant_excerpt(
                query,
                text,
                max_chars=_MAX_RERANK_TEXT_CHARS,
            )
        )
    return "\n".join(part for part in parts if part).strip() or addr.location
