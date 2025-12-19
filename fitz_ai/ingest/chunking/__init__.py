# ingest/chunking/__init__.py
"""
Chunking subsystem for ingest.

Provides:
- Chunker plugins
- ChunkingEngine
- Registry + auto-discovery
"""

from .base import BaseChunker, Chunk, ChunkerPlugin
from .engine import ChunkingEngine

__all__ = ["BaseChunker", "ChunkerPlugin", "ChunkingEngine", "Chunk"]
