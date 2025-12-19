"""
Vector database failures for fitz_ingest.
Used when communicating with Qdrant during ingestion.
"""

from __future__ import annotations

from fitz_ai.ingest.exceptions.base import IngestionError


class IngestionVectorError(IngestionError):
    """Failures in Qdrant-related operations (collection creation, upsert, etc)."""

    pass
