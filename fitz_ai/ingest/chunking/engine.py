# fitz_ai/ingest/chunking/engine.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.base import ChunkerPlugin
from fitz_ai.ingest.chunking.registry import get_chunking_plugin
from fitz_ai.ingest.config.schema import ChunkerConfig
from fitz_ai.ingest.exceptions.chunking import IngestionChunkingError
from fitz_ai.ingest.exceptions.config import IngestionConfigError
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import CHUNKING

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
            PluginCls = get_chunking_plugin(cfg.plugin_name)
        except Exception as e:
            raise IngestionConfigError(
                f"Unknown chunker plugin {cfg.plugin_name!r}"
            ) from e

        # Simple: just pass all kwargs to the plugin
        # The plugin defines what parameters it accepts
        kwargs: Dict[str, Any] = dict(cfg.kwargs or {})

        try:
            plugin = PluginCls(**kwargs)
        except Exception as e:
            raise IngestionConfigError(
                f"Failed to initialize chunker plugin {cfg.plugin_name!r}"
            ) from e

        return cls(plugin)

    def run(self, raw_doc: object) -> List[Chunk]:
        """
        Chunk a raw document into smaller pieces.

        Args:
            raw_doc: A RawDocument object with 'path', 'content', and 'metadata' attributes.
                     The 'content' attribute contains the already-extracted text.

        Returns:
            List of Chunk objects.
        """
        # Get the path for logging/metadata purposes
        path_str = getattr(raw_doc, "path", "unknown")
        path = Path(path_str) if path_str != "unknown" else None

        logger.debug(f"{CHUNKING} Chunking document: {path_str}")

        # IMPORTANT: Use the content from raw_doc, NOT re-read from disk!
        # The ingestion plugin has already extracted the text (including PDF text extraction).
        text = getattr(raw_doc, "content", None)

        if text is None:
            # Fallback: try to read from disk (for backwards compatibility)
            logger.warning(
                f"{CHUNKING} raw_doc has no 'content' attribute, falling back to disk read"
            )
            if path is None or not path.exists() or not path.is_file():
                raise IngestionChunkingError(
                    f"File does not exist or is not a file: {path_str}"
                )
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                logger.error(f"{CHUNKING} Failed reading file '{path}': {e}")
                raise IngestionChunkingError(
                    f"Failed reading file for chunking: {path}"
                ) from e

        if not text or not text.strip():
            logger.warning(f"{CHUNKING} Empty content for document: {path_str}")
            return []

        # Build base metadata
        base_meta: Dict[str, Any] = {
            "source_file": str(path) if path else path_str,
            **(getattr(raw_doc, "metadata", None) or {}),
        }

        try:
            chunks = self.plugin.chunk_text(text, base_meta)
        except IngestionChunkingError:
            raise
        except Exception as e:
            logger.error(f"{CHUNKING} Chunking plugin failed for '{path_str}': {e}")
            raise IngestionChunkingError(
                f"Chunking plugin failed for file '{path_str}'"
            ) from e

        logger.debug(f"{CHUNKING} Extracted {len(chunks)} chunks from '{path_str}'")
        return chunks
