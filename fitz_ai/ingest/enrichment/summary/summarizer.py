# fitz_ai/ingest/enrichment/summary/summarizer.py
"""
Chunk summarizer for generating searchable descriptions.

The summarizer:
1. Takes chunk content and context
2. Builds a prompt based on content type
3. Generates a description using an LLM
4. Caches results to avoid redundant API calls

Supports batch summarization to reduce API calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from fitz_ai.ingest.enrichment.base import (
    CodeEnrichmentContext,
    ContentType,
    EnrichmentContext,
)
from fitz_ai.ingest.enrichment.context.registry import get_context_registry
from fitz_ai.ingest.enrichment.summary.cache import SummaryCache

logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a chunk to summarize."""

    content: str
    file_path: str
    content_hash: str


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


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

    def summarize_batch(
        self,
        chunks: list[ChunkInfo],
        batch_size: int = 20,
    ) -> list[str]:
        """
        Generate descriptions for multiple chunks efficiently.

        Batches chunks into single LLM calls to reduce API overhead.
        Uses cache for already-summarized chunks.

        Args:
            chunks: List of ChunkInfo objects to summarize
            batch_size: Max chunks per LLM call (default 20)

        Returns:
            List of descriptions in same order as input chunks
        """
        if not chunks:
            return []

        results: list[str | None] = [None] * len(chunks)
        to_summarize: list[tuple[int, ChunkInfo]] = []

        # Check cache first
        for i, chunk in enumerate(chunks):
            cached = self._cache.get(chunk.content_hash, self.enricher_id)
            if cached is not None:
                results[i] = cached
            else:
                to_summarize.append((i, chunk))

        if not to_summarize:
            logger.info(f"[SUMMARIZE] All {len(chunks)} chunks cached")
            return results  # type: ignore

        logger.info(
            f"[SUMMARIZE] {len(chunks) - len(to_summarize)}/{len(chunks)} cached, "
            f"summarizing {len(to_summarize)} chunks"
        )

        # Process in batches
        for batch_start in range(0, len(to_summarize), batch_size):
            batch = to_summarize[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(to_summarize) + batch_size - 1) // batch_size

            logger.info(
                f"[SUMMARIZE] Batch {batch_num}/{total_batches}: {len(batch)} chunks"
            )

            descriptions = self._summarize_batch_single(batch)

            for (orig_idx, chunk), desc in zip(batch, descriptions):
                results[orig_idx] = desc
                self._cache.set(chunk.content_hash, self.enricher_id, desc)

        return results  # type: ignore

    def _summarize_batch_single(
        self,
        batch: list[tuple[int, ChunkInfo]],
    ) -> list[str]:
        """Summarize a single batch of chunks in one LLM call."""
        # Build combined prompt
        prompt_parts = [
            "You are generating descriptions for code chunks for a search index. "
            "Each description will be embedded and used to match developer questions to relevant code.\n\n"
            "For each numbered chunk below, write a 2-3 sentence description that explains:\n"
            "1. What the code does\n"
            "2. When a developer would look for it\n\n"
            "Format your response as numbered descriptions matching the chunk numbers.\n"
            "Example format:\n"
            "[1] This function handles user authentication by validating credentials against the database.\n"
            "[2] This class manages database connections and provides connection pooling.\n\n"
            "CHUNKS TO DESCRIBE:\n"
        ]

        for i, (_, chunk) in enumerate(batch, 1):
            file_name = Path(chunk.file_path).name
            # Truncate content to avoid token limits
            content = (
                chunk.content[:1500] if len(chunk.content) > 1500 else chunk.content
            )
            prompt_parts.append(f"\n--- CHUNK [{i}] from {file_name} ---\n{content}\n")

        prompt_parts.append(
            "\n--- END OF CHUNKS ---\n\nNow provide numbered descriptions:"
        )

        prompt = "".join(prompt_parts)
        messages = [{"role": "user", "content": prompt}]

        response = self._chat.chat(messages)

        # Parse numbered descriptions from response
        descriptions = self._parse_batch_response(response, len(batch))

        return descriptions

    def _parse_batch_response(self, response: str, expected_count: int) -> list[str]:
        """Parse numbered descriptions from LLM response."""
        # Match patterns like [1], [2], etc. or 1., 2., etc.
        pattern = r"(?:\[(\d+)\]|^(\d+)\.)\s*(.+?)(?=(?:\[\d+\]|^\d+\.|\Z))"
        matches = re.findall(pattern, response, re.MULTILINE | re.DOTALL)

        descriptions: dict[int, str] = {}
        for match in matches:
            num_str = match[0] or match[1]
            desc = match[2].strip()
            if num_str:
                num = int(num_str)
                if 1 <= num <= expected_count:
                    descriptions[num] = desc

        # If regex parsing failed, try line-by-line
        if len(descriptions) < expected_count:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                # Try to match [N] or N. at start
                m = re.match(r"(?:\[(\d+)\]|(\d+)\.)\s*(.+)", line)
                if m:
                    num_str = m.group(1) or m.group(2)
                    desc = m.group(3).strip()
                    if num_str:
                        num = int(num_str)
                        if 1 <= num <= expected_count and num not in descriptions:
                            descriptions[num] = desc

        # Build result list, using fallback for missing descriptions
        result = []
        for i in range(1, expected_count + 1):
            if i in descriptions:
                result.append(descriptions[i])
            else:
                logger.warning(
                    f"[SUMMARIZE] Missing description for chunk {i}, using fallback"
                )
                result.append("Code chunk - see source for details.")

        return result

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
            ".js": "JavaScript",
            ".jsx": "JavaScript/React",
            ".ts": "TypeScript",
            ".tsx": "TypeScript/React",
            ".java": "Java",
            ".kt": "Kotlin",
            ".go": "Go",
            ".rs": "Rust",
            ".c": "C",
            ".cpp": "C++",
            ".rb": "Ruby",
            ".php": "PHP",
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


__all__ = ["ChunkSummarizer", "ChunkInfo", "ChatClient"]
