# fitz_rag/pipeline/standard.py
from __future__ import annotations

from dataclasses import dataclass

from fitz_rag.config.schema import RAGConfig
from fitz_rag.pipeline.engine import RAGPipeline


@dataclass
class StandardRAG:
    """
    Official balanced preset.
    No legacy parameters.
    No config mutation outside a shallow clone.

    Usage:
        rag = StandardRAG(config)
        rag.ask("question")
    """

    config: RAGConfig
    pipeline: RAGPipeline

    def __init__(self, config: RAGConfig):
        # Optionally insert standard tuning here.
        # For now: no changes. Keep pure.
        self.config = config
        self.pipeline = RAGPipeline.from_config(config)

    def ask(self, query: str):
        return self.pipeline.run(query)
