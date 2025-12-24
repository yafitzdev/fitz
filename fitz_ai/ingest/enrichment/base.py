# fitz_ai/ingest/enrichment/base.py
"""
Base types for the enrichment system.

This module defines extensible base classes and protocols:
- EnrichmentContext: Base context (extend for new content types)
- CodeEnrichmentContext: Extended context for code files
- ContextBuilder: Protocol for building context from files
- Enricher: Protocol for generating descriptions

Design Philosophy:
    The enrichment system is designed to be content-type agnostic.
    Different content types (Python, JavaScript, Markdown, PDF, etc.)
    can provide different levels of context for enrichment.

    Content types with rich structure (like Python with imports/exports)
    provide more context, resulting in better descriptions.
    Content types without structure still get basic descriptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class ContentType(str, Enum):
    """Content type classification for routing."""

    PYTHON = "python"
    CODE = "code"  # Non-Python code
    DOCUMENT = "document"  # Markdown, text, etc.
    STRUCTURED = "structured"  # JSON, YAML, etc.
    UNKNOWN = "unknown"


@dataclass
class EnrichmentContext:
    """
    Base context for enrichment.

    This is the minimal context provided to the enricher.
    Extend this class for content types that can provide richer context.

    Attributes:
        file_path: Absolute path to the source file
        content_type: Classification of the content
        file_extension: File extension (e.g., ".py", ".md")
        metadata: Additional metadata (extensible)
    """

    file_path: str
    content_type: ContentType = ContentType.UNKNOWN
    file_extension: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.file_extension:
            self.file_extension = Path(self.file_path).suffix.lower()


@dataclass
class CodeEnrichmentContext(EnrichmentContext):
    """
    Extended context for code files.

    Provides structural information about code that helps
    generate more accurate descriptions.

    Attributes:
        language: Programming language (e.g., "python", "javascript")
        imports: List of imports/dependencies
        exports: List of exported symbols (classes, functions, etc.)
        used_by: List of (file_path, role) tuples - who imports this file
        docstring: Module/file-level docstring if present
    """

    language: str = ""
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    used_by: list[tuple[str, str]] = field(default_factory=list)  # (file, role)
    docstring: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.content_type == ContentType.UNKNOWN:
            self.content_type = ContentType.CODE


@dataclass
class DocumentEnrichmentContext(EnrichmentContext):
    """
    Context for document files (markdown, text, etc.).

    Can be extended later with document-specific metadata
    like headings, sections, links, etc.

    Attributes:
        title: Document title if extractable
        headings: List of headings in the document
    """

    title: str | None = None
    headings: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.content_type == ContentType.UNKNOWN:
            self.content_type = ContentType.DOCUMENT


@runtime_checkable
class ContextBuilder(Protocol):
    """
    Protocol for building enrichment context from files.

    Implement this protocol to add support for new content types.
    Each builder declares which file extensions it handles.

    Example:
        class PythonContextBuilder:
            supported_extensions = {".py", ".pyw"}

            def build(self, file_path: str, content: str) -> CodeEnrichmentContext:
                # Parse Python AST, extract imports/exports
                ...

        class MarkdownContextBuilder:
            supported_extensions = {".md", ".markdown"}

            def build(self, file_path: str, content: str) -> DocumentEnrichmentContext:
                # Extract title, headings
                ...
    """

    supported_extensions: set[str]

    def build(self, file_path: str, content: str) -> EnrichmentContext:
        """
        Build enrichment context for a file.

        Args:
            file_path: Path to the file
            content: File content (already parsed/read)

        Returns:
            EnrichmentContext (or subclass) with extracted information
        """
        ...


@runtime_checkable
class Enricher(Protocol):
    """
    Protocol for generating descriptions from chunks.

    The enricher takes a chunk and its context, and generates
    a natural language description suitable for embedding/search.

    Attributes:
        enricher_id: Unique identifier for cache invalidation
                     (e.g., "llm:openai:gpt-4o-mini:v1")
    """

    enricher_id: str

    def enrich(
        self,
        content: str,
        context: EnrichmentContext,
    ) -> str:
        """
        Generate a searchable description for content.

        Args:
            content: The chunk content to describe
            context: Context information about the source

        Returns:
            Natural language description (2-4 sentences typically)
        """
        ...


__all__ = [
    "ContentType",
    "EnrichmentContext",
    "CodeEnrichmentContext",
    "DocumentEnrichmentContext",
    "ContextBuilder",
    "Enricher",
]
