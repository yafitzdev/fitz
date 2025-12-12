# ingest/chunker/__init__.py
"""
Chunking subsystem for ingest.

Provides:
- Chunker plugins
- ChunkingEngine
- Registry + auto-discovery
"""

from .base import BaseChunker, ChunkerPlugin, Chunk
from .engine import ChunkingEngine

__all__ = ["BaseChunker", "ChunkerPlugin", "ChunkingEngine", "Chunk"]
