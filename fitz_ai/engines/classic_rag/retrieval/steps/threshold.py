# fitz_ai/engines/classic_rag/retrieval/steps/threshold.py
"""
Threshold Step - Filter chunks by score threshold.

Removes chunks below the threshold based on rerank or vector score.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep

logger = get_logger(__name__)


@dataclass
class ThresholdStep(RetrievalStep):
    """
    Filter chunks by score threshold.

    Removes chunks below the threshold. Uses rerank_score if available,
    otherwise falls back to vector_score.

    Args:
        threshold: Minimum score to keep a chunk (default: 0.5)
        score_key: Primary score key to check (default: "rerank_score")
        fallback_key: Fallback score key if primary not found (default: "vector_score")
    """

    threshold: float = 0.5
    score_key: str = "rerank_score"  # Which score to use
    fallback_key: str = "vector_score"  # Fallback if primary not found

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        logger.debug(
            f"{RETRIEVER} ThresholdStep: τ={self.threshold}, input={len(chunks)}"
        )

        filtered: list[Chunk] = []
        for chunk in chunks:
            score = chunk.metadata.get(self.score_key)
            if score is None:
                score = chunk.metadata.get(self.fallback_key)
            if score is None:
                # No score - include by default
                filtered.append(chunk)
                continue

            if score >= self.threshold:
                filtered.append(chunk)

        logger.debug(f"{RETRIEVER} ThresholdStep: output={len(filtered)} chunks")
        return filtered
