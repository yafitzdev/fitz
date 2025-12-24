# fitz_ai/ingest/enrichment/router.py
"""
Enrichment router for generating searchable descriptions.

The router:
1. Routes chunks to appropriate context builders based on file type
2. Generates descriptions using an LLM
3. Caches results to avoid redundant API calls

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
    │ (full ctx)   │   │ (basic ctx)  │   │  (minimal)   │
    └──────────────┘   └──────────────┘   └──────────────┘

Extensibility:
    To add support for a new content type:
    1. Create a ContextBuilder implementation
    2. Register it with the router via register_builder()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Set, runtime_checkable

from fitz_ai.ingest.enrichment.base import (
    ContentType,
    ContextBuilder,
    EnrichmentContext,
)
from fitz_ai.ingest.enrichment.cache import EnrichmentCache

logger = logging.getLogger(__name__)


# Default code extensions (non-Python)
DEFAULT_CODE_EXTENSIONS: Set[str] = {
    ".js", ".jsx", ".ts", ".tsx",  # JavaScript/TypeScript
    ".java", ".kt", ".scala",       # JVM
    ".go",                          # Go
    ".rs",                          # Rust
    ".c", ".cpp", ".h", ".hpp",     # C/C++
    ".cs",                          # C#
    ".rb",                          # Ruby
    ".php",                         # PHP
    ".swift",                       # Swift
    ".m", ".mm",                    # Objective-C
}

# Default document extensions
DEFAULT_DOCUMENT_EXTENSIONS: Set[str] = {
    ".md", ".markdown", ".rst",     # Markup
    ".txt", ".text",                # Plain text
    ".pdf",                         # PDF
    ".html", ".htm",                # HTML
    ".xml",                         # XML
    ".json", ".yaml", ".yml",       # Data formats
    ".toml", ".ini", ".cfg",        # Config files
}


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def complete(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        ...


class EnrichmentRouter:
    """
    Routes chunks to appropriate enrichment strategies.

    The router maintains a registry of context builders for different
    file types. When enriching a chunk, it:
    1. Selects the appropriate context builder
    2. Builds context for the chunk
    3. Generates a description using the LLM
    4. Caches the result

    Usage:
        router = EnrichmentRouter(
            chat_client=my_llm_client,
            cache=EnrichmentCache(cache_path),
            enricher_id="llm:gpt-4o-mini:v1",
        )

        # Register context builders
        router.register_builder(python_builder)

        # Enrich a chunk
        description = router.enrich(
            content="def hello(): ...",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )
    """

    def __init__(
        self,
        *,
        chat_client: ChatClient,
        cache: EnrichmentCache,
        enricher_id: str,
        code_extensions: Set[str] | None = None,
        document_extensions: Set[str] | None = None,
    ):
        """
        Initialize the router.

        Args:
            chat_client: LLM client for generating descriptions
            cache: Cache for storing generated descriptions
            enricher_id: Unique identifier for cache invalidation
            code_extensions: Extensions to treat as code (non-Python)
            document_extensions: Extensions to treat as documents
        """
        self._chat = chat_client
        self._cache = cache
        self.enricher_id = enricher_id

        self._code_extensions = code_extensions or DEFAULT_CODE_EXTENSIONS
        self._document_extensions = document_extensions or DEFAULT_DOCUMENT_EXTENSIONS

        # Registry of context builders by extension
        self._builders: Dict[str, ContextBuilder] = {}

    def register_builder(self, builder: ContextBuilder) -> None:
        """
        Register a context builder for specific extensions.

        Args:
            builder: The context builder to register
        """
        for ext in builder.supported_extensions:
            ext_lower = ext.lower()
            if ext_lower in self._builders:
                logger.warning(
                    f"Overwriting existing builder for {ext_lower}"
                )
            self._builders[ext_lower] = builder
            logger.debug(f"Registered builder for {ext_lower}")

    def enrich(
        self,
        content: str,
        file_path: str,
        content_hash: str,
    ) -> str:
        """
        Generate a searchable description for content.

        Args:
            content: The chunk content to describe
            file_path: Path to the source file
            content_hash: Hash of the content (for caching)

        Returns:
            Natural language description suitable for embedding
        """
        # Check cache first
        cached = self._cache.get(content_hash, self.enricher_id)
        if cached is not None:
            logger.debug(f"Cache hit for {file_path}")
            return cached

        # Build context
        context = self._build_context(file_path, content)

        # Generate description
        prompt = self._build_prompt(content, context)
        description = self._chat.complete(prompt)

        # Cache result
        self._cache.set(content_hash, self.enricher_id, description)

        logger.debug(f"Generated description for {file_path}")
        return description

    def _build_context(self, file_path: str, content: str) -> EnrichmentContext:
        """Build appropriate context for the file type."""
        ext = Path(file_path).suffix.lower()

        # Check for registered builder
        if ext in self._builders:
            return self._builders[ext].build(file_path, content)

        # Fall back to default contexts
        if ext in self._code_extensions:
            return EnrichmentContext(
                file_path=file_path,
                content_type=ContentType.CODE,
            )
        elif ext in self._document_extensions:
            return EnrichmentContext(
                file_path=file_path,
                content_type=ContentType.DOCUMENT,
            )
        else:
            return EnrichmentContext(
                file_path=file_path,
                content_type=ContentType.UNKNOWN,
            )

    def _build_prompt(self, content: str, context: EnrichmentContext) -> str:
        """Build the LLM prompt based on content type."""
        if context.content_type == ContentType.PYTHON:
            return self._build_python_prompt(content, context)
        elif context.content_type == ContentType.CODE:
            return self._build_code_prompt(content, context)
        elif context.content_type == ContentType.DOCUMENT:
            return self._build_document_prompt(content, context)
        else:
            return self._build_generic_prompt(content, context)

    def _build_python_prompt(self, content: str, context: EnrichmentContext) -> str:
        """Build prompt for Python code with full context."""
        from fitz_ai.ingest.enrichment.base import CodeEnrichmentContext

        if not isinstance(context, CodeEnrichmentContext):
            return self._build_code_prompt(content, context)

        # Format imports
        imports_str = ", ".join(context.imports[:10]) if context.imports else "none"
        if len(context.imports) > 10:
            imports_str += f" (+{len(context.imports) - 10} more)"

        # Format exports
        exports_str = ", ".join(context.exports[:10]) if context.exports else "none"
        if len(context.exports) > 10:
            exports_str += f" (+{len(context.exports) - 10} more)"

        # Format used_by
        if context.used_by:
            used_by_parts = [
                f"{Path(f).name} ({role})" for f, role in context.used_by[:5]
            ]
            used_by_str = ", ".join(used_by_parts)
            if len(context.used_by) > 5:
                used_by_str += f" (+{len(context.used_by) - 5} more)"
        else:
            used_by_str = "none detected"

        return f"""You are describing Python code for a search index. Your description will be embedded and used to match developer questions to relevant code.

File: {context.file_path}
Depends on: {imports_str}
Exports: {exports_str}
Used by: {used_by_str}

Code:
```python
{content}
```

Write 2-4 sentences describing:
1. What this code does (its purpose)
2. Its role in the system (based on imports/exports/usage)
3. When a developer would look for this code

Use natural language that matches how developers ask questions. Be specific about functionality, not vague.
Do NOT just restate the code - explain the purpose and context."""

    def _build_code_prompt(self, content: str, context: EnrichmentContext) -> str:
        """Build prompt for non-Python code."""
        # Try to infer language from extension
        ext = context.file_extension
        lang_map = {
            ".js": "JavaScript", ".jsx": "JavaScript/React",
            ".ts": "TypeScript", ".tsx": "TypeScript/React",
            ".java": "Java", ".kt": "Kotlin",
            ".go": "Go", ".rs": "Rust",
            ".c": "C", ".cpp": "C++",
            ".rb": "Ruby", ".php": "PHP",
        }
        language = lang_map.get(ext, "code")

        return f"""You are describing {language} code for a search index. Your description will be embedded and used to match developer questions to relevant code.

File: {context.file_path}

Code:
```
{content}
```

Write 2-4 sentences describing:
1. What this code does
2. When a developer would need this code

Use natural language that matches how developers ask questions. Be specific about functionality."""

    def _build_document_prompt(self, content: str, context: EnrichmentContext) -> str:
        """Build prompt for documents."""
        return f"""You are describing a document for a search index. Your description will be embedded and used to match user questions to relevant content.

File: {context.file_path}

Content:
{content}

Write 2-3 sentences summarizing:
1. What this document is about
2. What questions it would answer

Use natural language. Be specific about the topics covered."""

    def _build_generic_prompt(self, content: str, context: EnrichmentContext) -> str:
        """Build prompt for unknown content types."""
        return f"""You are describing content for a search index. Your description will be embedded and used to match user questions to relevant content.

File: {context.file_path}

Content:
{content}

Write 2-3 sentences summarizing what this content is about and when someone would look for it."""

    def save_cache(self) -> None:
        """Explicitly save the cache to disk."""
        self._cache.save()


class EnrichmentRouterBuilder:
    """
    Builder for creating EnrichmentRouter instances.

    Provides a fluent API for configuring the router.

    Usage:
        router = (
            EnrichmentRouterBuilder()
            .with_chat_client(client)
            .with_cache(cache)
            .with_enricher_id("llm:gpt-4o-mini:v1")
            .with_python_support(analyzer)
            .build()
        )
    """

    def __init__(self):
        self._chat_client: ChatClient | None = None
        self._cache: EnrichmentCache | None = None
        self._enricher_id: str | None = None
        self._builders: List[ContextBuilder] = []
        self._code_extensions: Set[str] | None = None
        self._document_extensions: Set[str] | None = None

    def with_chat_client(self, client: ChatClient) -> "EnrichmentRouterBuilder":
        """Set the chat client for LLM calls."""
        self._chat_client = client
        return self

    def with_cache(self, cache: EnrichmentCache) -> "EnrichmentRouterBuilder":
        """Set the cache for storing descriptions."""
        self._cache = cache
        return self

    def with_enricher_id(self, enricher_id: str) -> "EnrichmentRouterBuilder":
        """Set the enricher ID for cache invalidation."""
        self._enricher_id = enricher_id
        return self

    def with_builder(self, builder: ContextBuilder) -> "EnrichmentRouterBuilder":
        """Add a context builder."""
        self._builders.append(builder)
        return self

    def with_python_support(
        self,
        analyzer: Any,  # PythonProjectAnalyzer
    ) -> "EnrichmentRouterBuilder":
        """Add Python support with full context."""
        from fitz_ai.ingest.enrichment.python_context import PythonContextBuilder
        self._builders.append(PythonContextBuilder(analyzer))
        return self

    def with_code_extensions(self, extensions: Set[str]) -> "EnrichmentRouterBuilder":
        """Set custom code extensions."""
        self._code_extensions = extensions
        return self

    def with_document_extensions(self, extensions: Set[str]) -> "EnrichmentRouterBuilder":
        """Set custom document extensions."""
        self._document_extensions = extensions
        return self

    def build(self) -> EnrichmentRouter:
        """Build the router."""
        if self._chat_client is None:
            raise ValueError("chat_client is required")
        if self._cache is None:
            raise ValueError("cache is required")
        if self._enricher_id is None:
            raise ValueError("enricher_id is required")

        router = EnrichmentRouter(
            chat_client=self._chat_client,
            cache=self._cache,
            enricher_id=self._enricher_id,
            code_extensions=self._code_extensions,
            document_extensions=self._document_extensions,
        )

        for builder in self._builders:
            router.register_builder(builder)

        return router


__all__ = [
    "EnrichmentRouter",
    "EnrichmentRouterBuilder",
    "ChatClient",
    "DEFAULT_CODE_EXTENSIONS",
    "DEFAULT_DOCUMENT_EXTENSIONS",
]
