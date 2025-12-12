"""
Chunking subsystem for fitz_ingest.

This package provides:

- BaseChunker / ChunkerPlugin: interface for chunking plugins
- ChunkingEngine: orchestration and file I/O wrapper
- registry: auto-discovery of plugins in fitz_ingest.chunker.plugins
"""

from .base import BaseChunker, ChunkerPlugin, Chunk
from .engine import ChunkingEngine

__all__ = ["BaseChunker", "ChunkerPlugin", "ChunkingEngine", "Chunk"]
