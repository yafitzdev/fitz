# fitz_ai/engines/classic_rag/retrieval/steps/threshold.py
"""
Threshold Step - Filter chunks by score threshold.

Removes chunks below the threshold based on rerank or vector score.
VIP chunks (score=1.0) bypass the threshold and are always included.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import RetrievalStep

logger = get_logger(__name__)


def _is_vip(chunk: Chunk) -> bool:
    """Check if chunk has VIP status (score=1.0, bypasses threshold)."""
    meta = chunk.metadata
    return meta.get("rerank_score") == 1.0 or meta.get("score") == 1.0


@dataclass
class ThresholdStep(RetrievalStep):
    """
    Filter chunks by score threshold.

    VIP chunks (score=1.0) bypass the threshold and are always included.
    Regular chunks below threshold are filtered, but a minimum number
    of regular chunks are always kept regardless of score.

    Args:
        threshold: Minimum score to keep a chunk (default: 0.5)
        score_key: Primary score key to check (default: "rerank_score")
        fallback_key: Fallback score key if primary not found (default: "vector_score")
        min_chunks: Minimum regular chunks to keep regardless of threshold (default: 3)
    """

    threshold: float = 0.5
    score_key: str = "rerank_score"
    fallback_key: str = "vector_score"
    min_chunks: int = 3

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        logger.debug(
            f"{RETRIEVER} ThresholdStep: τ={self.threshold}, min={self.min_chunks}, input={len(chunks)}"
        )

        # Separate VIP from regular chunks
        vip: list[Chunk] = []
        regular: list[Chunk] = []
        for chunk in chunks:
            if _is_vip(chunk):
                vip.append(chunk)
            else:
                regular.append(chunk)

        # Filter regular chunks by threshold
        above_threshold: list[Chunk] = []
        below_threshold: list[Chunk] = []
        for chunk in regular:
            score = chunk.metadata.get(self.score_key)
            if score is None:
                score = chunk.metadata.get(self.fallback_key)
            if score is None:
                # No score - include by default
                above_threshold.append(chunk)
                continue

            if score >= self.threshold:
                above_threshold.append(chunk)
            else:
                below_threshold.append(chunk)

        # Ensure minimum regular chunks (take from below_threshold if needed)
        result_regular = above_threshold
        if len(result_regular) < self.min_chunks and below_threshold:
            needed = self.min_chunks - len(result_regular)
            # Take top N from below_threshold (already sorted by score)
            result_regular.extend(below_threshold[:needed])

        logger.debug(
            f"{RETRIEVER} ThresholdStep: vip={len(vip)}, regular={len(result_regular)}"
        )

        return vip + result_regular
