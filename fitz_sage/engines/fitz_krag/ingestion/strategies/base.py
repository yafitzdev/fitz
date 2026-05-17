# fitz_sage/engines/fitz_krag/ingestion/strategies/base.py
"""Base protocol and data types for ingestion strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class SymbolEntry:
    """A code symbol extracted from source."""

    name: str
    qualified_name: str
    kind: str  # "module", "function", "class", "method", "constant"
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str = ""  # author docstring / doc-comment, folded into the symbol FTS index
    imports: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)


@dataclass
class ImportEdge:
    """An import relationship between files."""

    target_module: str
    import_names: list[str] = field(default_factory=list)


@dataclass
class IngestResult:
    """Result of extracting symbols and imports from a source file."""

    symbols: list[SymbolEntry] = field(default_factory=list)
    imports: list[ImportEdge] = field(default_factory=list)


class IngestStrategy(Protocol):
    """Protocol for language-specific ingestion strategies."""

    def content_types(self) -> set[str]:
        """Return file extensions this strategy handles (e.g., {'.py'})."""
        ...

    def extract(self, source: str, file_path: str) -> IngestResult:
        """Extract symbols and imports from source code."""
        ...
