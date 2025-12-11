# fitz_rag/pipeline/fast.py
from __future__ import annotations

from dataclasses import dataclass

from fitz_rag.config.schema import RAGConfig, RerankConfig
from fitz_rag.pipeline.engine import RAGPipeline


@dataclass
class FastRAG:
    """
    Low-latency preset:
        - disables reranking
        - otherwise identical to config-defined pipeline

    Usage:
        rag = FastRAG(config)
        rag.ask("question")
    """

    config: RAGConfig
    pipeline: RAGPipeline

    def __init__(self, config: RAGConfig):
        # Clone config with modifications
        cfg = config.copy()
        cfg.rerank = RerankConfig(
            provider=config.rerank.provider,
            model=config.rerank.model,
            api_key=config.rerank.api_key,
            enabled=False,  # <--- Fast mode
        )

        self.config = cfg
        self.pipeline = RAGPipeline.from_config(cfg)

    def ask(self, query: str):
        return self.pipeline.run(query)
