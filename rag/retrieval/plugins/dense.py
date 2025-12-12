# rag/retrieval/plugins/dense.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, List, Mapping, Protocol, Sequence, runtime_checkable

from rag.exceptions.retriever import EmbeddingError, RerankError, VectorSearchError
from rag.models.chunk import Chunk
from rag.retrieval.base import RetrievalPlugin

from core.llm.rerank.engine import RerankEngine
from core.logging.logger import get_logger
from core.logging.tags import RETRIEVER

logger = get_logger(__name__)


Vector = Sequence[float]


@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> Vector: ...


@runtime_checkable
class VectorSearchHit(Protocol):
    id: Any
    payload: Mapping[str, Any] | None
    score: float | None


@runtime_checkable
class VectorSearchClient(Protocol):
    def search(
        self,
        collection_name: str,
        query_vector: Vector,
        limit: int,
        with_payload: bool = True,
    ) -> list[Any]:
        ...


@runtime_checkable
class DenseRetrieverConfig(Protocol):
    collection: str
    top_k: int


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    """
    Dense vector retrieval plugin.

    Contracts:
    - client: VectorSearchClient
    - retriever_cfg: DenseRetrieverConfig
    - embedder: Embedder
    - emits canonical Chunk objects
    """

    plugin_name: ClassVar[str] = "dense"

    client: VectorSearchClient | None = None
    retriever_cfg: DenseRetrieverConfig | None = None
    embedder: Embedder | None = None
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
            hits = self.client.search(
                collection_name=self.retriever_cfg.collection,
                query_vector=query_vector,
                limit=self.retriever_cfg.top_k,
                with_payload=True,
            )
        except TypeError:
            try:
                hits = self.client.search(  # type: ignore[call-arg]
                    self.retriever_cfg.collection,
                    query_vector,
                    self.retriever_cfg.top_k,
                )
            except Exception as exc:
                raise VectorSearchError("Vector search failed") from exc
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
