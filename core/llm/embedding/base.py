# core/llm/embedding/base.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingPlugin(Protocol):
    """
    Canonical embedding plugin contract.

    Provider-specific logic must live in plugin implementations only.
    """

    def embed(self, text: str) -> list[float]:
        ...
