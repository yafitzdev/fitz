from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from rag.models.chunk import Chunk
from rag.exceptions.retriever import (
    EmbeddingError,
    RerankError,
    VectorSearchError,
)
from rag.config.schema import EmbeddingConfig, RetrieverConfig, RerankConfig
from rag.retrieval.base import RetrievalPlugin

from core.logging.logger import get_logger
from core.logging.tags import RETRIEVER

from core.llm.embedding.engine import EmbeddingEngine
from core.llm.rerank.engine import RerankEngine
from core.llm.registry import get_llm_plugin

logger = get_logger(__name__)


def _config_to_kwargs(cfg: Any) -> dict:
    if cfg is None:
        return {}

    if hasattr(cfg, "model_dump"):
        data = cfg.model_dump(exclude_none=True)
    elif hasattr(cfg, "dict"):
        data = cfg.dict(exclude_none=True)
    elif isinstance(cfg, dict):
        data = {k: v for k, v in cfg.items() if v is not None}
    else:
        data = {
            k: v for k, v in vars(cfg).items()
            if not k.startswith("_") and v is not None
        }

    data.pop("plugin_name", None)
    return data


@dataclass
class DenseRetrievalPlugin(RetrievalPlugin):
    plugin_name: str = "dense"

    client: Any | None = None
    embed_cfg: EmbeddingConfig | None = None
    retriever_cfg: RetrieverConfig | None = None
    rerank_cfg: RerankConfig | None = None

    embedder: EmbeddingEngine | None = None
    rerank_engine: RerankEngine | None = None

    def __post_init__(self) -> None:
        if self.retriever_cfg is None:
            raise ValueError("retriever_cfg must be provided")

        if self.embedder is None and self.embed_cfg is None:
            raise ValueError("embed_cfg must be provided when no embedder is injected")

        if self.embedder is None:
            if not getattr(self.embed_cfg, "plugin_name", None):
                raise ValueError("embed_cfg.plugin_name must be set")

            EmbedCls = get_llm_plugin(
                plugin_name=self.embed_cfg.plugin_name,
                plugin_type="embedding",
            )
            embed_plugin = EmbedCls(**_config_to_kwargs(self.embed_cfg))
            self.embedder = EmbeddingEngine(embed_plugin)

        if (
            self.rerank_cfg
            and getattr(self.rerank_cfg, "enabled", False)
            and self.rerank_engine is None
        ):
            if not getattr(self.rerank_cfg, "plugin_name", None):
                raise ValueError("rerank_cfg.plugin_name must be set")

            RerankCls = get_llm_plugin(
                plugin_name=self.rerank_cfg.plugin_name,
                plugin_type="rerank",
            )
            rerank_plugin = RerankCls(**_config_to_kwargs(self.rerank_cfg))
            self.rerank_engine = RerankEngine(rerank_plugin)

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

            chunk = Chunk(
                id=str(getattr(hit, "id", idx)),
                doc_id=str(
                    payload.get("doc_id")
                    or payload.get("document_id")
                    or payload.get("source")
                    or "unknown"
                ),
                content=payload.get("content", payload.get("text", "")),
                metadata={
                    **payload,
                    "score": getattr(hit, "score", None),
                },
                chunk_index=int(payload.get("chunk_index", idx)),
            )

            chunks.append(chunk)

        if self.rerank_engine:
            try:
                chunks = self.rerank_engine.plugin.rerank(query, chunks)
            except Exception as exc:
                raise RerankError("Reranking failed") from exc

        return chunks


@dataclass
class RAGRetriever(DenseRetrievalPlugin):
    pass
