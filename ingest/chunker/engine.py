# ingest/chunker/engine.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from ingest.chunker.base import BaseChunker
import ingest.chunker.registry as chunker_registry
from ingest.config.schema import ChunkerConfig
from ingest.exceptions.config import IngestionConfigError
from ingest.exceptions.chunking import IngestionChunkingError

from core.logging.logger import get_logger
from core.logging.tags import CHUNKING

logger = get_logger(__name__)


class ChunkingEngine:
    """
    Central controller for all chunking operations.

    Supports:
    - plugin-style chunkers (chunk_text)
    - engine-style chunkers (run)
    """

    def __init__(self, plugin):
        self.plugin = plugin

    # ---------------------------------------------------------
    # Factory
    # ---------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: ChunkerConfig) -> "ChunkingEngine":
        if not cfg or not cfg.plugin_name:
            raise IngestionConfigError("ChunkerConfig.plugin_name is required")

        try:
            PluginFactory = chunker_registry.get_chunker_plugin(cfg.plugin_name)
        except Exception as e:
            raise IngestionConfigError(
                f"Unknown chunker plugin '{cfg.plugin_name}'"
            ) from e

        kwargs = {
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            **(cfg.options or {}),
        }

        try:
            plugin = PluginFactory(**kwargs)
        except Exception as e:
            raise IngestionConfigError(
                f"Failed to initialize chunker plugin '{cfg.plugin_name}'"
            ) from e

        return cls(plugin)

    # ---------------------------------------------------------
    # Public API (ENGINE BOUNDARY)
    # ---------------------------------------------------------
    def run(self, raw_doc) -> List[Dict[str, Any]]:
        """
        Chunk a RawDocument or engine-compatible object.
        """

        # Engine-style chunker (tests, injected engines)
        if hasattr(self.plugin, "run"):
            return self.plugin.run(raw_doc)

        # Plugin-style chunker
        path = Path(raw_doc.path)

        logger.debug(f"{CHUNKING} Chunking file: {path}")

        if not path.exists() or not path.is_file():
            logger.error(f"{CHUNKING} File does not exist or is not a file: {path}")
            return []

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"{CHUNKING} Failed reading file '{path}': {e}")
            raise IngestionConfigError(
                f"Failed reading file for chunking: {path}"
            ) from e

        base_meta: Dict[str, Any] = {
            "source_file": str(path),
            **(raw_doc.metadata or {}),
        }

        try:
            chunks = self.plugin.chunk_text(text, base_meta)
        except IngestionChunkingError:
            raise
        except Exception as e:
            logger.error(f"{CHUNKING} Chunking plugin failed for '{path}': {e}")
            raise IngestionChunkingError(
                f"Chunking plugin failed for file '{path}'"
            ) from e

        logger.debug(f"{CHUNKING} Extracted {len(chunks)} chunks from '{path}'")
        return chunks
