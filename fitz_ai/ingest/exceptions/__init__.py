"""
Unified import surface for all fitz_ingest exceptions.
"""

from .base import IngestionError
from .chunking import IngestionChunkingError
from .config import IngestionConfigError
from .vector import IngestionVectorError

__all__ = [
    "IngestionError",
    "IngestionConfigError",
    "IngestionVectorError",
    "IngestionChunkingError",
]
