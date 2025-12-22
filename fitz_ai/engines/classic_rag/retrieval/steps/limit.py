# fitz_ai/engines/classic_rag/retrieval/steps/limit.py
"""
Limit Step - Truncate to final k chunks.

Simple truncation assuming chunks are already sorted by relevance.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep

logger = get_logger(__name__)


@dataclass
class LimitStep(RetrievalStep):
    """
    Limit output to final k chunks.

    Simple truncation - assumes chunks are already sorted by relevance.

    Args:
        k: Maximum number of chunks to return (default: 5)
    """

    k: int = 5

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        logger.debug(f"{RETRIEVER} LimitStep: k={self.k}, input={len(chunks)}")
        return chunks[: self.k]