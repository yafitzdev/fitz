# fitz_ai/ingest/enrichment/context/__init__.py
"""
Context builder system for chunk enrichment.

Context builders extract structural information from files to provide
rich context for LLM-based summary generation. Each builder handles
specific file types and extracts relevant metadata (imports, exports,
headings, etc.).

Plugins are auto-discovered from the plugins/ directory.
"""

from fitz_ai.ingest.enrichment.context.registry import (
    ContextPluginInfo,
    ContextRegistry,
    get_context_plugin,
    get_context_registry,
    list_context_plugins,
)

__all__ = [
    "ContextRegistry",
    "ContextPluginInfo",
    "get_context_registry",
    "get_context_plugin",
    "list_context_plugins",
]
