# fitz/retrieval/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from fitz.core.models.chunk import Chunk
from fitz.retrieval.runtime.base import RetrievalPlugin
from fitz.retrieval.runtime.registry import get_retriever_plugin


@dataclass(slots=True)
class RetrieverEngine:
    """
    Thin orchestration layer around a retrieval plugin.

    Responsibilities:
    - Choose a retrieval plugin by name (via registry)
    - Construct plugin instance with caller-provided dependencies
    - Delegate `retrieve` calls
    """

    plugin: RetrievalPlugin

    @classmethod
    def from_name(cls, name: str, **plugin_kwargs: Any) -> "RetrieverEngine":
        plugin_cls = get_retriever_plugin(name)
        plugin = plugin_cls(**plugin_kwargs)  # type: ignore[arg-type]
        return cls(plugin=plugin)

    def retrieve(self, query: str) -> List[Chunk]:
        return self.plugin.retrieve(query)
