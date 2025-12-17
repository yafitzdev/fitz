# core/llm/rerank/engine.py
from __future__ import annotations

from typing import List

from fitz.core.llm.rerank.base import RerankPlugin
from fitz.engines.classic_rag.models.chunk import Chunk


class RerankEngine:
    """
    Thin wrapper around a rerank plugin.

    Architecture:
    - plugin construction is done upstream (pipeline wiring)
    - engine only enforces the contract and delegates calls
    """

    def __init__(self, plugin: RerankPlugin):
        self._plugin = plugin

    @property
    def plugin(self) -> RerankPlugin:
        return self._plugin

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        out = self._plugin.rerank(query, chunks)
        if not isinstance(out, list) or any(not isinstance(c, Chunk) for c in out):
            raise TypeError("RerankPlugin.rerank must return List[Chunk]")
        return out
