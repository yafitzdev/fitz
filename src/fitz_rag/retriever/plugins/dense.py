from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any

from fitz_rag.core import Chunk  # universal chunk

# Embedding
from fitz_rag.llm.embedding.engine import EmbeddingEngine
from fitz_rag.llm.embedding.plugins.cohere import CohereEmbeddingClient

# Reranking
from fitz_rag.llm.rerank.engine import RerankEngine
from fitz_rag.llm.rerank.plugins.cohere import CohereRerankClient

# Exceptions
from fitz_rag.exceptions.retriever import (
    EmbeddingError,
    VectorSearchError,
    RerankError,
)

# Config types
from fitz_rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig

# Logging
from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import RETRIEVER, EMBEDDING, VECTOR_SEARCH, RERANK

from fitz_rag.retriever.base import RetrievalPlugin

logger = get_logger(__name__)


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    """
    Dense vector retrieval plugin for fitz-rag.

    Rules:
      - retriever_cfg MUST be provided
      - embed_cfg is ONLY required if embedder is NOT injected
    """

    plugin_name: str = "dense"

    client: Any = None
    embed_cfg: EmbeddingConfig | None = None
    retriever_cfg: RetrieverConfig | None = None
    rerank_cfg: RerankConfig | None = None

    embedder: Any | None = None
    rerank_engine: RerankEngine | None = None

    def __post_init__(self) -> None:

        # ------------------------------
        # 1. retriever_cfg MUST exist
        # ------------------------------
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")

        # ------------------------------
        # 2. Embedding engine creation
        # ------------------------------
        if self.embedder is None:
            if self.embed_cfg is None:
                raise ValueError(
                    "embed_cfg must be provided when no embedder is injected"
                )

            embed_plugin = CohereEmbeddingClient(
                api_key=self.embed_cfg.api_key,
                model=self.embed_cfg.model,
                input_type="search_query",
                output_dimension=self.embed_cfg.output_dimension,
            )
            self.embedder = EmbeddingEngine(embed_plugin)

        # ------------------------------
        # 3. Rerank engine (optional)
        # ------------------------------
        if self.rerank_cfg and self.rerank_cfg.enabled:
            rerank_plugin = CohereRerankClient(
                api_key=self.rerank_cfg.api_key,
                model=self.rerank_cfg.model,
            )
            self.rerank_engine = RerankEngine(rerank_plugin)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        # -----------------------------
        # 1) EMBED QUERY
        # -----------------------------
        try:
            logger.debug(f"{EMBEDDING} Embedding query...")
            query_vector = self.embedder.embed(query)
        except Exception as e:
            logger.error(f"{EMBEDDING} Embedding failed for query='{query}'")
            raise EmbeddingError(f"Failed to embed query: {query}") from e

        # -----------------------------
        # 2) VECTOR SEARCH
        # -----------------------------
        try:
            logger.debug(
                f"{VECTOR_SEARCH} Qdrant search → collection='{self.retriever_cfg.collection}', "
                f"top_k={self.retriever_cfg.top_k}"
            )

            try:
                hits = self.client.search(
                    collection_name=self.retriever_cfg.collection,
                    query_vector=query_vector,
                    limit=self.retriever_cfg.top_k,
                    with_payload=True,
                )
            except TypeError:
                hits = self.client.search(
                    self.retriever_cfg.collection,
                    query_vector,
                    self.retriever_cfg.top_k,
                )

        except Exception as e:
            logger.error(f"{VECTOR_SEARCH} Search failed: {e}")
            raise VectorSearchError(
                f"Vector search failed for collection '{self.retriever_cfg.collection}'"
            ) from e

        # -----------------------------
        # 3) Convert → Chunk objects
        # -----------------------------
        chunks: List[Chunk] = []

        for hit in hits:
            payload = getattr(hit, "payload", {}) or {}

            chunks.append(
                Chunk(
                    id=getattr(hit, "id", None),
                    text=payload.get("text", ""),
                    metadata=payload,
                    score=getattr(hit, "score", None),
                )
            )

        # -----------------------------
        # 4) RERANK (optional)
        # -----------------------------
        if self.rerank_engine:
            logger.debug(f"{RERANK} Running rerank engine on {len(chunks)} chunks")
            try:
                chunks = self.rerank_engine.plugin.rerank(query, chunks)
            except Exception as e:
                logger.error(f"{RERANK} Rerank failed: {e}")
                raise RerankError("Reranking failed") from e

        return chunks


@dataclass
class RAGRetriever(DenseRetrievalPlugin):
    pass
