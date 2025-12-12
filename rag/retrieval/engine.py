# rag/retrieval/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from rag.models.chunk import Chunk
from rag.retrieval.base import RetrievalPlugin
from rag.retrieval.registry import get_retriever_plugin


@dataclass
class RetrieverEngine:
    """
    Thin orchestration layer around a retrieval plugin.

    Responsibilities:
    - Choose a retrieval plugin by name (via registry)
    - Hold plugin instance and delegate `retrieve` calls
    - Provide a future extension point for:
        * hooks / middleware
        * query rewriting
        * multi-step retrieval
        * logging / metrics around retrieval
    """

    plugin: RetrievalPlugin

    @classmethod
    def from_name(cls, name: str, **plugin_kwargs: Any) -> "RetrieverEngine":
        """
        Construct a RetrieverEngine from a registered plugin name
        and keyword arguments required by that plugin.

        Example:
            engine = RetrieverEngine.from_name(
                "dense",
                client=qdrant_client,
                embed_cfg=embed_cfg,
                retriever_cfg=retr_cfg,
                rerank_cfg=rerank_cfg,
            )
        """
        plugin_cls = get_retriever_plugin(name)
        plugin = plugin_cls(**plugin_kwargs)  # type: ignore[arg-type]
        return cls(plugin=plugin)

    def retrieve(self, query: str) -> List[Chunk]:
        """
        Delegate retrieval to the selected plugin.
        """
        return self.plugin.retrieve(query)
