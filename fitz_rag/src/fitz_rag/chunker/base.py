# src/fitz_rag/chunker/base.py
"""
Base chunker interface for fitz-rag.

A Chunker takes a file path and converts it into a list of Chunk objects,
with metadata. Each Chunk contains:
- id
- text
- metadata

The ingestion engine uses this to store data in Qdrant.
"""

from __future__ import annotations
from typing import List, Protocol
from fitz_rag.core import Chunk


class Chunker(Protocol):
    """
    Any chunker must implement chunk_file(path) -> List[Chunk].
    """

    def chunk_file(self, path: str) -> List[Chunk]:
        ...
