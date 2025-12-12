# rag/retrieval/plugins/dense.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, List

from rag.exceptions.retriever import EmbeddingError, RerankError, VectorSearchError
from rag.models.chunk import Chunk
from rag.retrieval.base import RetrievalPlugin

from core.llm.rerank.engine import RerankEngine
from core.logging.logger import get_logger
from core.logging.tags import RETRIEVER

logger = get_logger(__name__)


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    """
    Dense vector retrieval plugin.

    Architecture contracts:
    - Requires `client` and `retriever_cfg`.
    - Requires an embedder to be injected (engine responsibility).
    - Does NOT infer/resolve LLM plugins from config objects.
    - Emits canonical `Chunk` objects only.
    """

    plugin_name: ClassVar[str] = "dense"

    client: Any | None = None
    retriever_cfg: Any | None = None

    embedder: Any | None = None
    rerank_engine: RerankEngine | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            raise ValueError("client must be provided")
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")
        if self.embedder is None:
            raise ValueError("embedder must be injected (engine responsibility)")

    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        try:
            query_vector = self.embedder.embed(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {query}") from exc

        try:
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
        except Exception as exc:
            raise VectorSearchError("Vector search failed") from exc

        chunks: List[Chunk] = []

        for idx, hit in enumerate(hits):
            payload = getattr(hit, "payload", {}) or {}

            doc_id = (
                payload.get("doc_id")
                or payload.get("document_id")
                or payload.get("source")
                or "unknown"
            )
            chunk_index = payload.get("chunk_index", idx)
            content = payload.get("content", payload.get("text", ""))

            chunk_id = getattr(hit, "id", None)
            if chunk_id is None:
                chunk_id = f"{doc_id}:{chunk_index}"

            metadata = dict(payload)
            score = getattr(hit, "score", None)
            if score is not None:
                metadata["score"] = score

            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    doc_id=str(doc_id),
                    content=str(content),
                    chunk_index=int(chunk_index),
                    metadata=metadata,
                )
            )

        if self.rerank_engine is not None:
            try:
                chunks = self.rerank_engine.plugin.rerank(query, chunks)
            except Exception as exc:
                raise RerankError("Reranking failed") from exc

        return chunks


@dataclass
class RAGRetriever(DenseRetrievalPlugin):
    pass
