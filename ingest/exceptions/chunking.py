"""
Chunker and chunking-engine failures for fitz_ingest.
"""

from __future__ import annotations
from ingest.exceptions.base import IngestionError


class IngestionChunkingError(IngestionError):
    """Errors during text chunking (plugin failures, invalid chunk output, etc)."""
    pass
