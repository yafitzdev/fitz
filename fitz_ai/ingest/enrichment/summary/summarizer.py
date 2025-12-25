# fitz_ai/ingest/enrichment/summary/summarizer.py
"""
Chunk summarizer for generating searchable descriptions.

The summarizer:
1. Takes chunk content and context
2. Builds a prompt based on content type
3. Generates a description using an LLM
4. Caches results to avoid redundant API calls
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from fitz_ai.ingest.enrichment.base import (
    ContentType,
    EnrichmentContext,
    CodeEnrichmentContext,
)
from fitz_ai.ingest.enrichment.summary.cache import SummaryCache
from fitz_ai.ingest.enrichment.context.registry import get_context_registry

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str:
        ...


class ChunkSummarizer:
    """
    Generates searchable descriptions for chunks.

    Uses context builders to extract structural information from files,
    then generates natural language descriptions using an LLM.

    Usage:
        summarizer = ChunkSummarizer(
            chat_client=my_llm_client,
            cache=SummaryCache(cache_path),
            enricher_id="llm:gpt-4o-mini:v1",
        )

        description = summarizer.summarize(
            content="def hello(): ...",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )
    """

    def __init__(
        self,
        *,
        chat_client: ChatClient,
        cache: SummaryCache,
        enricher_id: str,
    ):
        self._chat = chat_client
        self._cache = cache
        self.enricher_id = enricher_id
        self._context_registry = get_context_registry()

    def summarize(
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
        cached = self._cache.get(content_hash, self.enricher_id)
        if cached is not None:
            logger.debug(f"Cache hit for {file_path}")
            return cached

        context = self._build_context(file_path, content)
        prompt = self._build_prompt(content, context)
        messages = [{"role": "user", "content": prompt}]
        description = self._chat.chat(messages)

        self._cache.set(content_hash, self.enricher_id, description)
        logger.debug(f"Generated description for {file_path}")

        return description

    def _build_context(self, file_path: str, content: str) -> EnrichmentContext:
        """Build appropriate context for the file type."""
        ext = Path(file_path).suffix.lower()

        plugin = self._context_registry.get_plugin_for_extension(ext)
        if plugin:
            builder = plugin.create_builder()
            return builder.build(file_path, content)

        # Fallback to generic context
        from fitz_ai.ingest.enrichment.context.plugins.generic import Builder
        return Builder().build(file_path, content)

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
        if not isinstance(context, CodeEnrichmentContext):
            return self._build_code_prompt(content, context)

        imports_str = ", ".join(context.imports[:10]) if context.imports else "none"
        if len(context.imports) > 10:
            imports_str += f" (+{len(context.imports) - 10} more)"

        exports_str = ", ".join(context.exports[:10]) if context.exports else "none"
        if len(context.exports) > 10:
            exports_str += f" (+{len(context.exports) - 10} more)"

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


__all__ = ["ChunkSummarizer", "ChatClient"]
