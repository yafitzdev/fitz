# fitz_ai/engines/classic_rag/retrieval/steps/dedupe.py
"""
Dedupe Step - Remove duplicate chunks.

Removes duplicates based on content, keeping first occurrence.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep

logger = get_logger(__name__)


@dataclass
class DedupeStep(RetrievalStep):
    """
    Remove duplicate chunks based on content.

    Keeps the first occurrence (assumes sorted by relevance).
    Comparison is case-insensitive with whitespace normalization.
    """

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        logger.debug(f"{RETRIEVER} DedupeStep: input={len(chunks)}")

        seen: set[str] = set()
        unique: list[Chunk] = []

        for chunk in chunks:
            # Normalize content for comparison
            key = chunk.content.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(chunk)

        logger.debug(f"{RETRIEVER} DedupeStep: output={len(unique)} chunks")
        return unique