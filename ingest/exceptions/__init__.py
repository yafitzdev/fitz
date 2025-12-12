"""
Unified import surface for all fitz_ingest exceptions.
"""

from .base import IngestionError
from .config import IngestionConfigError
from .vector import IngestionVectorError
from .chunking import IngestionChunkingError

__all__ = [
    "IngestionError",
    "IngestionConfigError",
    "IngestionVectorError",
    "IngestionChunkingError",
]
