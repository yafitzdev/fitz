# src/fitz_rag/core/types.py
"""
Core types and dataclasses shared across the fitz_rag library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Protocol, List, Optional


# ---------------------------------------------------------
# RetrievedChunk: common object returned by all strategies
# ---------------------------------------------------------
@dataclass
class RetrievedChunk:
    collection: str
    score: float
    text: str
    metadata: Dict[str, Any]
    chunk_id: Any


# ---------------------------------------------------------
# Chunk: generic chunk produced by a chunker
# ---------------------------------------------------------
@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


# ---------------------------------------------------------
# Chunker interface (Protocol)
# ---------------------------------------------------------
class Chunker(Protocol):
    """
    A Chunker takes an input path and produces a list of Chunk objects.
    """

    def chunk_file(self, path: str) -> List[Chunk]:
        ...
