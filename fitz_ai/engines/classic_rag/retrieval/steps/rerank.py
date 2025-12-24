# fitz_ai/engines/classic_rag/retrieval/steps/rerank.py
"""
Rerank Step - Reorder chunks using cross-encoder model.

Takes chunks from previous step, reranks by relevance, returns top-k.
VIP chunks (score=1.0) are excluded from reranking and always included.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.exceptions import RerankError
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import Reranker, RetrievalStep

logger = get_logger(__name__)


def _is_vip(chunk: Chunk) -> bool:
    """Check if chunk has VIP status (score=1.0, bypasses reranking)."""
    meta = chunk.metadata
    return meta.get("rerank_score") == 1.0 or meta.get("score") == 1.0


@dataclass
class RerankStep(RetrievalStep):
    """
    Rerank chunks using a cross-encoder or similar model.

    Takes top-k chunks from previous step, reranks them, returns top rerank_k.
    VIP chunks (score=1.0) are excluded from reranking and always prepended.

    Args:
        reranker: Reranking service
        k: Number of chunks to return after reranking (default: 10)
    """

    reranker: Reranker
    k: int = 10  # Return top k after reranking

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks

        logger.debug(f"{RETRIEVER} RerankStep: input={len(chunks)}, k={self.k}")

        # Separate VIP from regular chunks (VIP keep their score=1.0)
        vip: list[Chunk] = []
        regular_chunks: list[Chunk] = []
        for chunk in chunks:
            if _is_vip(chunk):
                vip.append(chunk)
            else:
                regular_chunks.append(chunk)

        if vip:
            logger.debug(f"{RETRIEVER} RerankStep: preserving {len(vip)} VIP chunks")

        if not regular_chunks:
            # Only VIP chunks, nothing to rerank
            return vip

        # Extract text for reranker
        documents = [chunk.content for chunk in regular_chunks]

        try:
            # Reranker returns [(index, score), ...] sorted by relevance
            ranked_results = self.reranker.rerank(query, documents, top_n=self.k)
        except Exception as exc:
            raise RerankError(f"Reranking failed: {exc}") from exc

        # Reorder chunks based on rerank results
        reranked: list[Chunk] = []
        for idx, score in ranked_results:
            if 0 <= idx < len(regular_chunks):
                chunk = regular_chunks[idx]
                # Add rerank score to metadata
                updated_metadata = dict(chunk.metadata)
                updated_metadata["rerank_score"] = score

                reranked.append(
                    Chunk(
                        id=chunk.id,
                        doc_id=chunk.doc_id,
                        content=chunk.content,
                        chunk_index=chunk.chunk_index,
                        metadata=updated_metadata,
                    )
                )

        logger.debug(f"{RETRIEVER} RerankStep: output={len(reranked)} chunks")

        # Prepend VIP chunks (they keep their original score=1.0)
        return vip + reranked
