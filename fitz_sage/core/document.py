# fitz_sage/core/document.py
"""
Core document types for the ingestion pipeline.

ParsedDocument represents structured content extracted from any file format.
It preserves document structure (headings, tables, code blocks, etc.) so
chunkers can make intelligent splitting decisions.

Flow: Source → Parser → ParsedDocument → Chunker → Chunks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass
class Table:
    """
    Structured table data extracted from a document.

    Used for CSV, Excel, SQLite tables, or tables extracted from PDFs.
    Stored separately in TableStore for SQL-like querying.
    """

    id: str  # Unique table identifier (e.g., "employees", "sales_2024")
    columns: List[str]  # Column headers
    rows: List[List[str]]  # Data rows (all values as strings)
    source_file: str  # Original file path
    metadata: Dict[str, Any] = field(default_factory=dict)  # Sheet name, DB name, etc.

    @property
    def row_count(self) -> int:
        """Number of data rows (excluding header)."""
        return len(self.rows)

    @property
    def column_count(self) -> int:
        """Number of columns."""
        return len(self.columns)

    def __repr__(self) -> str:
        return f"Table({self.id!r}, {self.column_count} cols, {self.row_count} rows)"


class ElementType(Enum):
    """Types of structural elements in a document."""

    TEXT = "text"  # Regular paragraph text
    HEADING = "heading"  # Section heading (h1-h6)
    TABLE = "table"  # Tabular data
    FIGURE = "figure"  # Image or diagram
    CODE_BLOCK = "code_block"  # Source code
    LIST_ITEM = "list_item"  # Bulleted or numbered list item
    QUOTE = "quote"  # Block quote


@dataclass
class DocumentElement:
    """
    A single structural element in a parsed document.

    Elements represent semantic units: paragraphs, headings, tables, etc.
    Chunkers use element boundaries to make better splitting decisions.
    """

    type: ElementType
    content: str
    level: Optional[int] = None  # Heading level (1-6), list depth, etc.
    language: Optional[str] = None  # Programming language for code blocks
    page: Optional[int] = None  # Page number (for PDFs)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"DocumentElement({self.type.value}, {preview!r})"


@dataclass
class ParsedDocument:
    """
    Structured representation of a parsed file.

    Produced by Parsers, consumed by Chunkers.
    Preserves document structure for intelligent chunking.
    """

    source: str  # Original source URI or path
    elements: List[DocumentElement]
    metadata: Dict[str, Any] = field(default_factory=dict)
    tables: List[Table] = field(default_factory=list)  # Structured tables (CSV, Excel, etc.)

    @property
    def full_text(self) -> str:
        """Get all text content concatenated (for simple chunkers)."""
        return "\n\n".join(el.content for el in self.elements if el.content)

    @property
    def element_count(self) -> int:
        """Number of elements in the document."""
        return len(self.elements)

    @property
    def page_count(self) -> Optional[int]:
        """Number of pages (if available from metadata)."""
        return self.metadata.get("page_count")

    def __repr__(self) -> str:
        return f"ParsedDocument({self.source!r}, {self.element_count} elements)"


__all__ = [
    "Table",
    "ElementType",
    "DocumentElement",
    "ParsedDocument",
]
