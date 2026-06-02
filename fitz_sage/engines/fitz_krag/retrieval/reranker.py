# fitz_sage/engines/fitz_krag/retrieval/reranker.py
"""
Address reranker for KRAG — cross-encoder reranking on address summaries.

Reranks retrieved addresses using a cross-encoder model before reading
their full content, improving precision of the top-k results.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

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
            return addresses[: self._k]

        documents = [_rerank_document(addr) for addr in addresses]

        try:
            ranked = self._reranker.rerank(query, documents, top_n=self._k)

            reranked: list[Address] = []
            for result in ranked:
                original = addresses[result.index]
                reranked.append(
                    Address(
                        kind=original.kind,
                        source_id=original.source_id,
                        location=original.location,
                        summary=original.summary,
                        score=result.score,
                        metadata=original.metadata,
                    )
                )

            logger.debug(f"Reranked {len(addresses)} addresses to top {len(reranked)}")
            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {e}")
            return addresses[: self._k]


def _rerank_document(addr: Address) -> str:
    """Build the text shown to the cross-encoder for one address."""
    parts = [addr.summary or addr.location]
    text = addr.metadata.get("text")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip()[:_MAX_RERANK_TEXT_CHARS])
    return "\n".join(part for part in parts if part).strip() or addr.location
