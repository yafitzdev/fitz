# core/llm/embedding/registry.py
from __future__ import annotations

from typing import Type

from fitz.core.llm.embedding.base import EmbeddingPlugin
from fitz.core.llm.registry import get_llm_plugin


def get_embedding_plugin(plugin_name: str) -> Type[EmbeddingPlugin]:
    """
    Return the embedding plugin class for the given plugin name.

    This is a thin type-safe alias over the central LLM registry.
    """
    return get_llm_plugin(plugin_name=plugin_name, plugin_type="embedding")  # type: ignore[return-value]
