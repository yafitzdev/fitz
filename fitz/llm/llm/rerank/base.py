# core/llm/rerank/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from fitz.core.models.chunk import Chunk


@runtime_checkable
class RerankPlugin(Protocol):
    plugin_name: str
    plugin_type: str  # must be "rerank"

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]: ...
