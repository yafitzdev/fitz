"""
Base exception types for fitz_ingest.

All ingestion-related exceptions inherit from IngestionError.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion-related exceptions."""

    pass
