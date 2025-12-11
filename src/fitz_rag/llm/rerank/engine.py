"""
Reranking engine for fitz-rag.

Separates reranking concerns from the retriever logic so that:
- Rerank plugins remain simple
- The engine handles logging, error wrapping, and ordering
- Multiple rerankers or score fusion can be added in v0.2.0
"""

from __future__ import annotations

from typing import List

from fitz_rag.core import Chunk
from fitz_rag.llm.rerank.plugins.cohere import CohereRerankClient
from fitz_rag.exceptions.retriever import RerankError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RERANK

logger = get_logger(__name__)


class RerankEngine:
    """
    Wraps a rerank plugin and produces an ordered list of chunks.
    """

    def __init__(self, plugin: any):
        self.plugin = plugin

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks or len(chunks) < 2:
            return chunks

        logger.debug(f"{RERANK} Running rerank on {len(chunks)} chunks")

        docs = [c.text for c in chunks]

        try:
            order = self.plugin.rerank(query, docs, top_n=len(docs))
        except Exception as e:
            logger.error(f"{RERANK} Reranker plugin failed: {e}")
            raise RerankError("Reranking failed") from e

        try:
            return [chunks[i] for i in order]
        except Exception as e:
            raise RerankError("Invalid rerank ordering returned by plugin") from e
