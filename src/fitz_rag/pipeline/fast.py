from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config
from fitz_rag.pipeline.engine import RAGPipeline


@dataclass
class FastRAG:
    """
    Fast (low-latency) RAG pipeline.

    Preferred usage:
        rag = FastRAG(config=my_cfg)

    Legacy usage:
        rag = FastRAG(collection="docs", top_k=5)
    """

    # Unified config (preferred)
    config: Optional[RAGConfig] = None

    # Legacy override fields
    collection: Optional[str] = None
    cohere_api_key: Optional[str] = None
    top_k: int = 5
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    pipeline: Optional[RAGPipeline] = None

    def __post_init__(self):
        # -------------------------------------
        # 1) Unified config path
        # -------------------------------------
        if self.config is not None:
            self.pipeline = RAGPipeline.from_config(self.config)
            return

        # -------------------------------------
        # 2) Legacy initialization path
        # -------------------------------------
        raw = load_config()

        # Apply minimal overrides
        raw["retriever"]["collection"] = self.collection
        raw["retriever"]["top_k"] = self.top_k

        # Disable reranking (fast mode)
        raw["rerank"]["enabled"] = False

        # API key passthrough
        if self.cohere_api_key:
            raw["llm"]["api_key"] = self.cohere_api_key
            raw["embedding"]["api_key"] = self.cohere_api_key
            raw["rerank"]["api_key"] = self.cohere_api_key

        # Build validated config
        cfg = RAGConfig.from_dict(raw)

        # Build pipeline from config
        self.pipeline = RAGPipeline.from_config(cfg)

    # -------------------------------------
    # User API
    # -------------------------------------
    def ask(self, query: str):
        return self.pipeline.run(query)
