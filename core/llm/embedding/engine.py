# core/llm/embedding/engine.py
from __future__ import annotations

from core.llm.embedding.base import EmbeddingPlugin


class EmbeddingEngine:
    """
    Thin wrapper around an embedding plugin.

    Architecture:
    - plugin construction is done upstream (pipeline wiring)
    - engine only enforces the contract and delegates calls
    """

    def __init__(self, plugin: EmbeddingPlugin):
        self._plugin = plugin

    @property
    def plugin(self) -> EmbeddingPlugin:
        return self._plugin

    def embed(self, text: str) -> list[float]:
        out = self._plugin.embed(text)
        if not isinstance(out, list) or any(not isinstance(x, (int, float)) for x in out):
            raise TypeError("EmbeddingPlugin.embed must return list[float]")
        return [float(x) for x in out]
