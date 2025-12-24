# fitz_ai/ingest/enrichment/__init__.py
"""
Enrichment module for generating searchable descriptions of chunks.

This module provides:
- EnrichmentContext: Context data for enrichment (extensible for different content types)
- ContextBuilder: Protocol for building context (implement for new content types)
- EnrichmentRouter: Routes chunks to appropriate enrichment strategies
- EnrichmentCache: Caches generated descriptions to avoid redundant LLM calls

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    EnrichmentRouter                          │
    │  Routes chunks to appropriate strategy based on content type │
    └─────────────────────────────────────────────────────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            ▼                   ▼                   ▼
    ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
    │   Python     │   │  Other Code  │   │   Documents  │
    │  (full ctx)  │   │ (basic ctx)  │   │  (minimal)   │
    └──────────────┘   └──────────────┘   └──────────────┘

Extensibility:
    To add support for a new content type:
    1. Create a new EnrichmentContext subclass (if needed)
    2. Implement a ContextBuilder for that type
    3. Register it with the router

Example:
    # Adding JavaScript support later:
    class JSContextBuilder(ContextBuilder):
        supported_extensions = {".js", ".ts", ".tsx"}

        def build(self, file_path: str, content: str) -> CodeEnrichmentContext:
            # Parse JS imports/exports
            ...
"""

from fitz_ai.ingest.enrichment.base import (
    ContentType,
    EnrichmentContext,
    CodeEnrichmentContext,
    DocumentEnrichmentContext,
    ContextBuilder,
    Enricher,
)
from fitz_ai.ingest.enrichment.cache import EnrichmentCache
from fitz_ai.ingest.enrichment.router import (
    EnrichmentRouter,
    EnrichmentRouterBuilder,
    ChatClient,
)
from fitz_ai.ingest.enrichment.python_context import (
    PythonProjectAnalyzer,
    PythonContextBuilder,
)

__all__ = [
    # Base types
    "ContentType",
    "EnrichmentContext",
    "CodeEnrichmentContext",
    "DocumentEnrichmentContext",
    "ContextBuilder",
    "Enricher",
    # Cache
    "EnrichmentCache",
    # Router
    "EnrichmentRouter",
    "EnrichmentRouterBuilder",
    "ChatClient",
    # Python support
    "PythonProjectAnalyzer",
    "PythonContextBuilder",
]
