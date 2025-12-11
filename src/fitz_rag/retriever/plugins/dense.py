from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any, Callable

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


def _run_embedder(embedder: Any, query: str):
    """
    Run the injected embedder in a DI-friendly way.

    Supported forms:
        - callable: embedder(query) -> vector
        - object with .embed(): embedder.embed(query) -> vector
    """
    # Callable embedder (e.g. lambda text: [...])
    if callable(embedder):
        return embedder(query)

    # Object with .embed()
    if hasattr(embedder, "embed") and callable(getattr(embedder, "embed")):
        return embedder.embed(query)

    raise TypeError("embedder must be callable or expose an .embed(query) method")


def _hit_to_chunk(hit: Any) -> Chunk:
    """
    Convert a Qdrant-like hit object into a universal Chunk.

    Expected hit attributes:
        - id
        - score
        - payload (dict with at least 'text')
    """
    payload = getattr(hit, "payload", {}) or {}

    return Chunk(
        id=getattr(hit, "id", None),
        text=payload.get("text", ""),
        metadata=payload,
        score=getattr(hit, "score", None),
    )


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    """
    Dense vector retrieval plugin for fitz-rag.

    Responsibilities:
      - Embed query
      - Perform dense vector search in Qdrant
      - Optionally rerank results
      - Return universal Chunk objects

    Dependency injection rules:
      - If `embedder` is provided, it is used AS-IS.
          * It may be a callable:   embedder(text) -> vector
          * Or an object with .embed(text) -> vector
      - Otherwise, a default EmbeddingEngine is built from `embed_cfg`.
      - If `rerank_engine` is provided, it's used AS-IS.
      - Otherwise, if `rerank_cfg.enabled` is True, a Cohere rerank engine
        is constructed (for now, Cohere is the only built-in reranker).
    """

    plugin_name: str = "dense"

    # Core dependencies
    client: Any = None
    embed_cfg: EmbeddingConfig | None = None
    retriever_cfg: RetrieverConfig | None = None
    rerank_cfg: RerankConfig | None = None

    # Injected test/prod dependencies
    # embedder can be:
    #   - callable: embedder(query) -> vector
    #   - object with .embed(query) -> vector
    embedder: Any | None = None

    # RerankEngine (or compatible) instance
    rerank_engine: RerankEngine | Any | None = None

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------
    def __post_init__(self) -> None:
        if self.embed_cfg is None or self.retriever_cfg is None:
            raise ValueError("embed_cfg and retriever_cfg must be provided")

        # 1. EMBEDDING ENGINE
        # If user injected embedder, DO NOT override it.
        if self.embedder is None:
            provider = (self.embed_cfg.provider or "").lower()
            if provider not in ("cohere", ""):
                # Provider not yet supported by the built-in default path.
                raise EmbeddingError(
                    f"Unsupported embedding provider for dense retriever: {self.embed_cfg.provider!r}"
                )

            embed_plugin = CohereEmbeddingClient(
                api_key=self.embed_cfg.api_key,
                model=self.embed_cfg.model,
                input_type="search_query",
                output_dimension=self.embed_cfg.output_dimension,
            )
            self.embedder = EmbeddingEngine(embed_plugin)

        # 2. RERANK ENGINE (if enabled and not injected)
        if self.rerank_engine is None and self.rerank_cfg and self.rerank_cfg.enabled:
            provider = (self.rerank_cfg.provider or "").lower()
            if provider not in ("cohere", ""):
                raise RerankError(
                    f"Unsupported rerank provider for dense retriever: {self.rerank_cfg.provider!r}"
                )

            rerank_plugin = CohereRerankClient(
                api_key=self.rerank_cfg.api_key,
                model=self.rerank_cfg.model,
            )
            self.rerank_engine = RerankEngine(rerank_plugin)

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")

        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        # -----------------------------
        # 1) EMBED QUERY
        # -----------------------------
        try:
            logger.debug(f"{EMBEDDING} Embedding query...")
            query_vector = _run_embedder(self.embedder, query)
        except Exception as e:
            logger.error(f"{EMBEDDING} Embedding failed for query='{query}': {e}")
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
                # Real Qdrant client (keyword args)
                hits = self.client.search(
                    collection_name=self.retriever_cfg.collection,
                    query_vector=query_vector,
                    limit=self.retriever_cfg.top_k,
                    with_payload=True,
                )
            except TypeError:
                # Mock / legacy clients (positional signature)
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
        # Convert hits → universal Chunk
        # -----------------------------
        chunks: List[Chunk] = []
        for hit in hits:
            chunks.append(_hit_to_chunk(hit))

        # -----------------------------
        # 3) OPTIONAL RERANKING
        # -----------------------------
        if self.rerank_engine:
            logger.debug(f"{RERANK} Running rerank engine on {len(chunks)} chunks")

            try:
                # Allow both:
                #   - RerankEngine(plugin=...) with .plugin.rerank()
                #   - Custom engines with .rerank()
                engine = self.rerank_engine
                if hasattr(engine, "plugin") and hasattr(engine.plugin, "rerank"):
                    chunks = engine.plugin.rerank(query, chunks)
                elif hasattr(engine, "rerank") and callable(engine.rerank):
                    chunks = engine.rerank(query, chunks)
                else:
                    raise TypeError(
                        "rerank_engine must expose .plugin.rerank(...) or .rerank(...)"
                    )
            except Exception as e:
                logger.error(f"{RERANK} Rerank failed: {e}")
                raise RerankError("Reranking failed") from e

        return chunks


# -------------------------------------------------------------------
# Backwards compatible alias used throughout the codebase & tests.
# -------------------------------------------------------------------
@dataclass
class RAGRetriever(DenseRetrievalPlugin):
    """
    Backwards-compatible alias for DenseRetrievalPlugin.

    Existing code can continue to import:

        from fitz_rag.retriever.plugins.dense import RAGRetriever
    """
    pass
