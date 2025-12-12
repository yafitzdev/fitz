from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from core.llm.rerank.base import RerankPlugin
from core.llm.registry import get_llm_plugin
from core.logging.logger import get_logger
from core.logging.tags import RERANK

logger = get_logger(__name__)


@dataclass
class RerankEngine:
    """
    Orchestration layer around a RerankPlugin.

    Responsibilities:
        - Call plugin.rerank(query, chunks)
        - Provide factory constructor from_name(...)
    """

    plugin: RerankPlugin

    def rerank(self, query: str, chunks: List[Any]) -> List[Any]:
        logger.info(f"{RERANK} Reranking {len(chunks)} chunks")
        indices = self.plugin.rerank(query, chunks)
        return [chunks[i] for i in indices]

    # ---------------------------------------------------------
    # Factory from registry
    # ---------------------------------------------------------
    @classmethod
    def from_name(cls, plugin_name: str, **kwargs) -> "RerankEngine":
        PluginCls = get_llm_plugin(plugin_name, plugin_type="rerank")
        plugin = PluginCls(**kwargs)
        return cls(plugin=plugin)
