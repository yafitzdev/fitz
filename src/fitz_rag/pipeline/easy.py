# fitz_rag/pipeline/easy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config
from fitz_rag.pipeline.engine import RAGPipeline


@dataclass
class EasyRAG:
    """
    Minimal convenience wrapper.

    Usage:
        rag = EasyRAG()                    # load default config
        rag = EasyRAG.from_path("cfg.yml") # load custom YAML
        rag = EasyRAG(config=my_cfg)       # provide typed config

        answer = rag.ask("What is RAG?")
    """

    config: RAGConfig
    pipeline: RAGPipeline

    # -------------------------
    # Constructors
    # -------------------------
    @classmethod
    def from_path(cls, path: str):
        raw = load_config(path)
        cfg = RAGConfig.from_dict(raw)
        return cls(config=cfg, pipeline=RAGPipeline.from_config(cfg))

    @classmethod
    def from_default(cls):
        raw = load_config()
        cfg = RAGConfig.from_dict(raw)
        return cls(config=cfg, pipeline=RAGPipeline.from_config(cfg))

    def __init__(self, config: Optional[RAGConfig] = None):
        if config is None:
            raw = load_config()
            config = RAGConfig.from_dict(raw)
        self.config = config
        self.pipeline = RAGPipeline.from_config(config)

    # -------------------------
    # User API
    # -------------------------
    def ask(self, query: str):
        return self.pipeline.run(query)
