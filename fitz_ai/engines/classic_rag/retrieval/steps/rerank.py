# fitz_ai/engines/classic_rag/retrieval/steps/rerank.py
"""
Rerank Step - Reorder chunks using cross-encoder model.

Takes chunks from previous step, reranks by relevance, returns top-k.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitz_ai.engines.classic_rag.exceptions import RerankError
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import RETRIEVER

from .base import Reranker, RetrievalStep

logger = get_logger(__name__)


@dataclass
class RerankStep(RetrievalStep):
    """
    Rerank chunks using a cross-encoder or similar model.

    Takes top-k chunks from previous step, reranks them, returns top rerank_k.

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

        # Separate artifacts from regular chunks (artifacts keep their score=1.0)
        artifacts: list[Chunk] = []
        regular_chunks: list[Chunk] = []
        for chunk in chunks:
            if chunk.metadata.get("is_artifact"):
                artifacts.append(chunk)
            else:
                regular_chunks.append(chunk)

        if artifacts:
            logger.debug(
                f"{RETRIEVER} RerankStep: preserving {len(artifacts)} artifacts"
            )

        if not regular_chunks:
            # Only artifacts, nothing to rerank
            return artifacts

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

        # Prepend artifacts (they keep their original score=1.0)
        return artifacts + reranked
