from __future__ import annotations

from typing import Protocol, Any

from rag.config.schema import RAGConfig


class PipelinePlugin(Protocol):
    """
    Minimal interface for pipeline plugins.

    A plugin takes a fully-resolved RAGConfig and returns a pipeline-like
    object with a .run(query: str) method (e.g. RAGPipeline or a wrapper).
    """

    def build(self, cfg: RAGConfig) -> Any:
        ...
