from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from core.llm.embedding.base import EmbeddingPlugin
from core.llm.registry import get_llm_plugin
from core.logging import get_logger
from core.logging_tags import EMBEDDING

logger = get_logger(__name__)


@dataclass
class EmbeddingEngine:
    """
    Orchestration for embedding plugins.

    Responsibilities:
        - Normalize input to a single text string
        - Run plugin.embed(...)
        - Provide factory constructor from_name(...)
    """

    plugin: EmbeddingPlugin

    def embed(self, text: str) -> List[float]:
        logger.info(f"{EMBEDDING} Embedding text of length={len(text)}")
        return self.plugin.embed(text)

    # ---------------------------------------------------------
    # Factory from registry
    # ---------------------------------------------------------
    @classmethod
    def from_name(cls, plugin_name: str, **kwargs) -> "EmbeddingEngine":
        PluginCls = get_llm_plugin(plugin_name, plugin_type="embedding")
        plugin = PluginCls(**kwargs)
        return cls(plugin=plugin)
