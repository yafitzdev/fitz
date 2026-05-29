# fitz_sage/ingestion/source/base.py
"""
SourceFile: the unit of input to a parser.

A SourceFile pairs an original URI with a locally-accessible path (plus
optional MIME type, size, and metadata). The ingestion pipeline constructs
these directly from discovered paths.

Flow: SourceFile → Parser.parse() → ParsedDocument
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SourceFile:
    """
    A file to be parsed.

    Provides unified access regardless of storage backend. The local_path is
    always available - remote files are downloaded to a temp location if needed.
    """

    uri: str  # Original URI (file://, s3://, mongodb://, etc.)
    local_path: Path  # Local path for parser access (may be temp file)
    mime_type: Optional[str] = None  # Detected or inferred MIME type
    size: Optional[int] = None  # File size in bytes
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        """File extension (lowercase, with dot)."""
        return self.local_path.suffix.lower()

    @property
    def name(self) -> str:
        """Filename without path."""
        return self.local_path.name

    def __repr__(self) -> str:
        return f"SourceFile({self.uri!r})"


__all__ = [
    "SourceFile",
]
