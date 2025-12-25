# fitz_ai/ingest/enrichment/context/plugins/generic.py
"""
Generic context builder plugin.

Provides basic context for files that don't have a specialized builder.
This is the fallback builder used when no other builder matches.
"""

from __future__ import annotations

from pathlib import Path

from fitz_ai.ingest.enrichment.base import (
    ContentType,
    EnrichmentContext,
)

plugin_name = "generic"
plugin_type = "context"
supported_extensions: set[str] = set()  # Empty = fallback for any extension


# Code file extensions (non-Python)
CODE_EXTENSIONS = {
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".scala",
    ".go", ".rs",
    ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php",
    ".swift", ".m", ".mm",
}

# Document extensions
DOCUMENT_EXTENSIONS = {
    ".md", ".markdown", ".rst",
    ".txt", ".text",
    ".html", ".htm",
}

# Structured data extensions
STRUCTURED_EXTENSIONS = {
    ".json", ".yaml", ".yml",
    ".toml", ".ini", ".cfg",
    ".xml",
}


class Builder:
    """
    Generic context builder for files without specialized builders.

    Provides basic context classification based on file extension.
    """

    supported_extensions = supported_extensions

    def build(self, file_path: str, content: str) -> EnrichmentContext:
        """Build basic enrichment context for any file."""
        ext = Path(file_path).suffix.lower()
        content_type = self._classify_content_type(ext)

        return EnrichmentContext(
            file_path=file_path,
            content_type=content_type,
            file_extension=ext,
        )

    def _classify_content_type(self, extension: str) -> ContentType:
        """Classify content type based on file extension."""
        if extension in CODE_EXTENSIONS:
            return ContentType.CODE
        elif extension in DOCUMENT_EXTENSIONS:
            return ContentType.DOCUMENT
        elif extension in STRUCTURED_EXTENSIONS:
            return ContentType.STRUCTURED
        else:
            return ContentType.UNKNOWN
