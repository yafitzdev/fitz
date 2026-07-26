# fitz_sage/ingestion/parser/base.py
"""
Parser protocol for document format understanding.

Parsers handle the "how" of extraction - converting raw file bytes into
structured ParsedDocument with preserved semantics (headings, tables, etc.)

Flow: SourceFile -> Parser.parse() -> ParsedDocument -> engine ingestion
"""

from __future__ import annotations

from typing import Protocol, Set, runtime_checkable

from fitz_sage.core.document import ParsedDocument
from fitz_sage.ingestion.source.base import SourceFile


@runtime_checkable
class Parser(Protocol):
    """
    Protocol for document parsers.

    Parsers understand specific file formats and extract structured content.
    They convert raw bytes into ParsedDocument with semantic elements.

    Example implementations:
    - DoclingParser: PDF, DOCX, images (via Docling library)
    - MarkdownParser: Markdown files
    - CodeParser: Source code (via tree-sitter)
    - PlainTextParser: Simple .txt files
    - WhisperParser: Audio transcription (future)
    """

    plugin_name: str
    supported_extensions: Set[str]  # e.g., {".pdf", ".docx", ".png"}

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a file into structured content.

        Args:
            file: SourceFile with local_path for reading.

        Returns:
            ParsedDocument with structured elements.

        Raises:
            ParseError: If the file cannot be parsed.
        """
        ...

    def can_parse(self, file: SourceFile) -> bool:
        """
        Check if this parser can handle the given file.

        Default implementation checks file extension.
        Override for more sophisticated detection (e.g., MIME type, magic bytes).

        Args:
            file: SourceFile to check.

        Returns:
            True if this parser can handle the file.
        """
        ...


class ParseError(Exception):
    """Raised when a parser fails to process a file."""

    def __init__(self, message: str, source: str, cause: Exception | None = None):
        super().__init__(message)
        self.source = source
        self.cause = cause


__all__ = [
    "Parser",
    "ParseError",
]
