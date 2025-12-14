# fitz/retrieval/plugins/dense.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Protocol, runtime_checkable

from fitz.core.exceptions.llm import EmbeddingError
from fitz.core.llm.embedding.engine import EmbeddingEngine
from fitz.core.llm.rerank.engine import RerankEngine
from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import RETRIEVER
from fitz.core.models.chunk import Chunk
from fitz.retrieval.exceptions.base import RerankError, VectorSearchError
from fitz.retrieval.runtime.base import RetrievalPlugin

logger = get_logger(__name__)


@runtime_checkable
class VectorSearchClient(Protocol):
    def search(self, *args: Any, **kwargs: Any) -> list[Any]: ...


@dataclass(frozen=True, slots=True)
class RetrieverCfg:
    collection: str
    top_k: int = 5


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    plugin_name: str = "dense"

    client: VectorSearchClient | None = None
    retriever_cfg: RetrieverCfg | None = None

    embedder: EmbeddingEngine | None = None
    rerank_engine: RerankEngine | None = None

    def __post_init__(self) -> None:
        if self.client is None:
            raise ValueError("client must be provided")
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")
        if self.embedder is None:
            raise ValueError("embedder must be injected (EmbeddingEngine)")

    def retrieve(self, query: str) -> List[Chunk]:
        logger.info(
            f"{RETRIEVER} Running retrieval for collection='{self.retriever_cfg.collection}'"
        )

        try:
            query_vector = self.embedder.embed(query)
        except Exception as exc:
            raise EmbeddingError(f"Failed to embed query: {query!r}") from exc

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
            payload = getattr(hit, "payload", None) or getattr(hit, "metadata", None) or {}
            if not isinstance(payload, dict):
                payload = {}

            chunk = Chunk(
                id=str(getattr(hit, "id", idx)),
                doc_id=str(
                    payload.get("doc_id")
                    or payload.get("document_id")
                    or payload.get("source")
                    or "unknown"
                ),
                content=str(payload.get("content") or payload.get("text") or ""),
                chunk_index=int(payload.get("chunk_index", idx)),
                metadata={
                    **payload,
                    "score": getattr(hit, "score", None),
                },
            )
            chunks.append(chunk)

        if self.rerank_engine:
            try:
                chunks = self.rerank_engine.rerank(query, chunks)
            except Exception as exc:
                raise RerankError("Reranking failed") from exc

        return chunks
