# core/llm/embedding/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPlugin(Protocol):
    plugin_name: str
    plugin_type: str  # must be "embedding"

    def embed(self, text: str) -> list[float]: ...
