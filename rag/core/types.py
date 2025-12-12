"""
Core types and dataclasses shared across the fitz_rag library.

This module now uses the universal Chunk model defined in fitz_stack.core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

# ---------------------------------------------------------
# Import the universal Chunk
# ---------------------------------------------------------
from core.core import Chunk


# ---------------------------------------------------------
# RetrievedChunk:
# A wrapper describing *how* a chunk was retrieved from the vector DB.
# NOTE: This is NOT the actual chunk object used in pipelines.
# ---------------------------------------------------------
@dataclass
class RetrievedChunk:
    collection: str
    score: float
    text: str
    metadata: Dict[str, Any]
    chunk_id: Any
