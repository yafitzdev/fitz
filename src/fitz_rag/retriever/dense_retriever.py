from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter

from fitz_rag.models.chunk import Chunk
from fitz_rag.config.schema import (
    RetrieverConfig,
    EmbeddingConfig,
    RerankConfig,
)
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient


class RAGRetriever:
    """
    Unified-config retriever.

    Handles:
      - embedding queries
      - vector search (Qdrant)
      - optional reranking
      - conversion to Chunk objects

    All parameters must come from the unified RAGConfig.
    """

    def __init__(
        self,
        client: QdrantClient,
        embed_cfg: EmbeddingConfig,
        retriever_cfg: RetrieverConfig,
        rerank_cfg: Optional[RerankConfig] = None,
    ):
        self.client = client
        self.retriever_cfg = retriever_cfg

        # -----------------------------
        # Embedding backend (Cohere)
        # -----------------------------
        if embed_cfg.provider.lower() != "cohere":
            raise ValueError(f"Unsupported embedding provider: {embed_cfg.provider}")
        self.embedder = CohereEmbeddingClient(
            api_key=embed_cfg.api_key,
            model=embed_cfg.model,
        )

        # -----------------------------
        # Optional reranker
        # -----------------------------
        if rerank_cfg and rerank_cfg.enabled:
            if rerank_cfg.provider.lower() != "cohere":
                raise ValueError(f"Unsupported rerank provider: {rerank_cfg.provider}")
            if not rerank_cfg.model:
                raise ValueError("RerankConfig missing model.")
            self.reranker = CohereRerankClient(
                api_key=rerank_cfg.api_key,
                model=rerank_cfg.model,
            )
        else:
            self.reranker = None

    # ---------------------------------------------------------------------
    # Main public API
    # ---------------------------------------------------------------------
    def retrieve(self, query: str) -> List[Chunk]:
        """
        Full retrieval pipeline:
           1) embed query
           2) vector search
           3) optional rerank
           4) convert to Chunk objects
        """
        # 1) Embed query
        query_vector = self.embedder.embed(query)

        # 2) Vector search
        top_k = self.retriever_cfg.top_k
        hits = self.client.search(
            collection_name=self.retriever_cfg.collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        # Convert to internal structure
        chunks = self._to_chunks(hits)

        # 3) Optional reranking
        if self.reranker is not None:
            chunks = self._apply_rerank(query, chunks)

        return chunks

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _to_chunks(self, hits) -> List[Chunk]:
        chunks: List[Chunk] = []

        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            metadata = payload.get("metadata", {})
            doc_id = payload.get("doc_id", "unknown")

            chunks.append(
                Chunk(
                    id=str(hit.id),
                    doc_id=doc_id,
                    content=text,
                    metadata=metadata,
                    chunk_index=metadata.get("chunk_index", 0),
                )
            )

        return chunks

    def _apply_rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Uses Cohere Rerank API on the retrieved chunks.
        """
        if not chunks:
            return chunks

        texts = [c.content for c in chunks]

        rerank_result = self.reranker.rerank(
            query=query,
            documents=texts,
        )

        # rerank_result.indices gives the new order
        ordered = [chunks[i] for i in rerank_result.indices]
        return ordered
