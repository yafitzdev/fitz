# core/llm/rerank/registry.py
from __future__ import annotations

from typing import Type

from fitz.core.llm.registry import get_llm_plugin
from fitz.core.llm.rerank.base import RerankPlugin


def get_rerank_plugin(plugin_name: str) -> Type[RerankPlugin]:
    """
    Return the rerank plugin class for the given plugin name.

    This is a thin type-safe alias over the central LLM registry.
    """
    return get_llm_plugin(plugin_name=plugin_name, plugin_type="rerank")  # type: ignore[return-value]
