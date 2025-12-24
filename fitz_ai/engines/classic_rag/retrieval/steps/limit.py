# fitz_ai/engines/classic_rag/retrieval/steps/limit.py
"""
Limit Step - Truncate to final k chunks.

Truncation assuming chunks are already sorted by relevance.
VIP chunks (score=1.0) are always included and don't count toward k.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep

logger = get_logger(__name__)


def _is_vip(chunk: Chunk) -> bool:
    """Check if chunk has VIP status (score=1.0, bypasses limits)."""
    meta = chunk.metadata
    return meta.get("rerank_score") == 1.0 or meta.get("score") == 1.0


@dataclass
class LimitStep(RetrievalStep):
    """
    Limit output to final k regular chunks.

    VIP chunks (score=1.0) are always included and excluded from limit.
    Only regular chunks compete for the k slots.

    Args:
        k: Maximum number of regular chunks to return (default: 5)
    """

    k: int = 5

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        # Separate VIP from regular chunks
        vip: list[Chunk] = []
        regular: list[Chunk] = []
        for chunk in chunks:
            if _is_vip(chunk):
                vip.append(chunk)
            else:
                regular.append(chunk)

        logger.debug(
            f"{RETRIEVER} LimitStep: k={self.k}, vip={len(vip)}, regular={len(regular)}"
        )

        # Limit only regular chunks
        limited = regular[: self.k]

        # Return VIP first, then limited regular chunks
        return vip + limited
