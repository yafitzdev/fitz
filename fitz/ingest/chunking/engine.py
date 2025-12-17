# ingest/chunking/engine.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fitz.logging.logger import get_logger
from fitz.logging.tags import CHUNKING
from fitz.engines.classic_rag.models.chunk import Chunk
from fitz.ingest.chunking.base import ChunkerPlugin
from fitz.ingest.chunking.registry import get_chunker_plugin
from fitz.ingest.config.schema import ChunkerConfig
from fitz.ingest.exceptions.chunking import IngestionChunkingError
from fitz.ingest.exceptions.config import IngestionConfigError

logger = get_logger(__name__)


class ChunkingEngine:
    """
    Central controller for chunking operations.

    Architecture:
    - Single contract: ChunkerPlugin.chunk_text(text, base_meta) -> list[Chunk]
    """

    def __init__(self, plugin: ChunkerPlugin):
        self.plugin = plugin

    @classmethod
    def from_config(cls, cfg: ChunkerConfig) -> "ChunkingEngine":
        if cfg is None or not getattr(cfg, "plugin_name", None):
            raise IngestionConfigError("ChunkerConfig.plugin_name is required")

        try:
            PluginCls = get_chunker_plugin(cfg.plugin_name)
        except Exception as e:
            raise IngestionConfigError(f"Unknown chunker plugin {cfg.plugin_name!r}") from e

        kwargs: Dict[str, Any] = {}
        if getattr(cfg, "chunk_size", None) is not None:
            kwargs["chunk_size"] = cfg.chunk_size
        if getattr(cfg, "chunk_overlap", None) is not None:
            kwargs["chunk_overlap"] = cfg.chunk_overlap
        if getattr(cfg, "options", None):
            kwargs.update(cfg.options)

        try:
            plugin = PluginCls(**kwargs)
        except Exception as e:
            raise IngestionConfigError(
                f"Failed to initialize chunker plugin {cfg.plugin_name!r}"
            ) from e

        return cls(plugin)

    def run(self, raw_doc: object) -> List[Chunk]:
        path = Path(raw_doc.path)

        logger.debug(f"{CHUNKING} Chunking file: {path}")

        if not path.exists() or not path.is_file():
            raise IngestionChunkingError(f"File does not exist or is not a file: {path}")

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"{CHUNKING} Failed reading file '{path}': {e}")
            raise IngestionChunkingError(f"Failed reading file for chunking: {path}") from e

        base_meta: Dict[str, Any] = {
            "source_file": str(path),
            **(getattr(raw_doc, "metadata", None) or {}),
        }

        try:
            chunks = self.plugin.chunk_text(text, base_meta)
        except IngestionChunkingError:
            raise
        except Exception as e:
            logger.error(f"{CHUNKING} Chunking plugin failed for '{path}': {e}")
            raise IngestionChunkingError(f"Chunking plugin failed for file '{path}'") from e

        logger.debug(f"{CHUNKING} Extracted {len(chunks)} chunks from '{path}'")
        return chunks
