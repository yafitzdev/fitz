# fitz_ai/ingest/chunking/__init__.py
"""
Chunking subsystem for ingestion.

Provides:
- ChunkerPlugin protocol for implementing chunkers
- ChunkingRouter for file-type specific routing
- ChunkingEngine for orchestration
- Built-in chunker plugins (simple, etc.)

Usage:
    from fitz_ai.ingest.chunking import (
        ChunkingRouter,
        ChunkingEngine,
        ChunkerPlugin,
    )
    from fitz_ai.engines.classic_rag.config import ChunkingRouterConfig

    config = ChunkingRouterConfig(...)
    router = ChunkingRouter.from_config(config)
    engine = ChunkingEngine(router)

    chunks = engine.run(raw_doc)
"""

from fitz_ai.ingest.chunking.base import Chunk, ChunkerPlugin
from fitz_ai.ingest.chunking.engine import ChunkingEngine
from fitz_ai.ingest.chunking.router import ChunkingRouter

__all__ = [
    # Protocol
    "ChunkerPlugin",
    "Chunk",
    # Router
    "ChunkingRouter",
    # Engine
    "ChunkingEngine",
]
