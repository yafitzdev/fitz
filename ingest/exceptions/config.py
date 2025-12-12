"""
Configuration and file/path errors for fitz_ingest.
"""

from __future__ import annotations
from ingest.exceptions.base import IngestionError


class IngestionConfigError(IngestionError):
    """Invalid arguments, bad file paths, unreadable files, or config issues."""
    pass
