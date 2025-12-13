# core/vector_db/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Type

from core.llm.registry import get_llm_plugin
from core.vector_db.base import SearchResult, VectorDBPlugin


@dataclass(slots=True)
class VectorDBEngine:
    plugin: VectorDBPlugin

    @classmethod
    def from_name(cls, plugin_name: str, **kwargs: Any) -> "VectorDBEngine":
        PluginCls: Type[Any] = get_llm_plugin(plugin_name=plugin_name, plugin_type="vector_db")
        plugin = PluginCls(**kwargs)
        return cls(plugin=plugin)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> List[SearchResult] | List[Any]:
        return self.plugin.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=with_payload,
        )
