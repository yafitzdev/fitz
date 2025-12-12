# core/llm/rerank/base.py
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from rag.models.chunk import Chunk


@runtime_checkable
class RerankPlugin(Protocol):
    """
    Canonical rerank plugin contract.

    Contract:
    - Input: list[Chunk]
    - Output: list[Chunk] (same objects reordered OR new Chunk objects)
    """

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        ...
