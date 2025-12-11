from __future__ import annotations

from dataclasses import dataclass
from typing import List

from fitz_rag.core import Chunk
from fitz_rag.llm.embedding_client import EmbeddingClient
from fitz_rag.llm.rerank.engine import RerankEngine
from fitz_rag.llm.rerank.plugins.cohere import CohereRerankClient

from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
)

from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RETRIEVER, EMBEDDING

logger = get_logger(__name__)


@dataclass
class RAGRetriever:
    """
    Dense vector retriever plugin (default retriever in fitz-rag).
    """

    client: any
    embed_cfg: EmbeddingConfig
    retriever_cfg: RetrieverConfig
    rerank_cfg: RerankConfig | None = None
    embedder: EmbeddingClient | None = None
    reranker: any = None
    rerank_engine: RerankEngine | None = None

    def __post_init__(self):
        from fitz_rag.llm.embedding_client import CohereEmbeddingClient

        if self.embedder is None:
            self.embedder = CohereEmbeddingClient(
                api_key=self.embed_cfg.api_key,
                model=self.embed_cfg.model,
            )

        # Build reranker plugin + engine
        if self.rerank_cfg and self.rerank_cfg.enabled:
            if self.reranker is None:
                self.reranker = CohereRerankClient(
                    api_key=self.rerank_cfg.api_key,
                    model=self.rerank_cfg.model,
                )
            self.rerank_engine = RerankEngine(self.reranker)

    # ---------------------------------------------------------
    # Main retrieval
    # ---------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        # 1. EMBEDDING
        try:
            logger.debug(f"{EMBEDDING} Embedding query...")
            query_vector = self.embedder.embed(query)
        except Exception as e:
            logger.error(f"{EMBEDDING} Embedding failed for query='{query}'")
            raise EmbeddingError(f"Failed to embed query: {query}") from e

        # 2. VECTOR SEARCH
        try:
            hits = self.client.search(
                collection_name=self.retriever_cfg.collection,
                vector=query_vector,
                limit=self.retriever_cfg.top_k,
                with_payload=True,
            )
        except Exception as e:
            raise VectorSearchError(
                f"Vector search failed for collection '{self.retriever_cfg.collection}'"
            ) from e

        chunks = [
            Chunk(
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata=hit.payload,
            )
            for hit in hits
        ]

        # 3. RERANK ENGINE
        if self.rerank_engine:
            chunks = self.rerank_engine.rerank(query, chunks)

        return chunks
