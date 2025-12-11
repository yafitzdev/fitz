from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from fitz_rag.core import Chunk
from fitz_rag.llm.rerank.base import RerankPlugin
from fitz_rag.llm.rerank.registry import get_rerank_plugin


@dataclass
class RerankEngine:
    """
    Thin orchestration layer for reranking plugins.

    Responsibilities:
    - Hold a plugin instance
    - Delegate rerank()
    - Support instantiation by name
    """

    plugin: RerankPlugin

    @classmethod
    def from_name(cls, name: str, **plugin_kwargs: Any) -> "RerankEngine":
        plugin_cls = get_rerank_plugin(name)
        plugin = plugin_cls(**plugin_kwargs)  # type: ignore
        return cls(plugin=plugin)

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        return self.plugin.rerank(query, chunks)
