from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config
from fitz_rag.pipeline.engine import RAGPipeline


@dataclass
class EasyRAG:
    """
    High-level convenience wrapper.

    Preferred usage:
        rag = EasyRAG(config=my_cfg)
        rag.ask("What happened?")

    Legacy usage still supported:
        rag = EasyRAG(collection="docs")
    """

    # OPTIONAL: unified config
    config: Optional[RAGConfig] = None

    # Legacy overrides (used only if config=None)
    collection: Optional[str] = None
    cohere_api_key: Optional[str] = None
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    top_k: int = 15
    final_top_k: int = 5

    pipeline: Optional[RAGPipeline] = None

    def __post_init__(self):
        # --------------------------
        # 1) Unified CONFIG path
        # --------------------------
        if self.config is not None:
            # Build full pipeline from config
            self.pipeline = RAGPipeline.from_config(self.config)
            return

        # --------------------------
        # 2) Legacy path (backward compatible)
        # --------------------------
        # Build RAGConfig dynamically from kwargs
        raw = load_config()

        # update necessary values
        raw["retriever"]["collection"] = self.collection
        raw["retriever"]["top_k"] = self.top_k

        if self.cohere_api_key:
            raw["llm"]["api_key"] = self.cohere_api_key
            raw["embedding"]["api_key"] = self.cohere_api_key
            raw["rerank"]["api_key"] = self.cohere_api_key

        cfg = RAGConfig.from_dict(raw)

        # Build pipeline
        self.pipeline = RAGPipeline.from_config(cfg)

    # ------------------------------------------------------------------
    # User API
    # ------------------------------------------------------------------
    def ask(self, query: str):
        return self.pipeline.run(query)
