# ============================
# File: src/fitz_stack/llm/__init__.py
# ============================
"""
Minimal init for fitz_stack.llm.

Do NOT import engine classes here to avoid circular imports.
Users should import from submodules directly:

    from fitz_stack.llm.embedding.engine import EmbeddingEngine
    from fitz_stack.llm.chat.engine import ChatEngine
    from fitz_stack.llm.rerank.engine import RerankEngine
"""

from .registry import (
    get_llm_plugin,
    register_llm_plugin,
    LLMRegistryError,
    PluginType,
)

__all__ = [
    "get_llm_plugin",
    "register_llm_plugin",
    "LLMRegistryError",
    "PluginType",
]
