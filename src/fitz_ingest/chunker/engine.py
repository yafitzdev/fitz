from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

from fitz_ingest.chunker.base import BaseChunker
from fitz_ingest.exceptions.config import IngestionConfigError
from fitz_ingest.exceptions.chunking import IngestionChunkingError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import CHUNKING

logger = get_logger(__name__)


class ChunkingEngine:
    """
    Central controller for all chunking operations.

    Responsibilities:
    - Logging
    - File reading
    - Error handling
    - Metadata injection
    - Delegation to chunking plugins (BaseChunker / ChunkerPlugin)

    The plugin only needs to implement:

        chunk_text(text: str, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]

    It does NOT need to handle:
    - file I/O
    - error handling
    - logging
    """

    def __init__(self, plugin: BaseChunker):
        self.plugin = plugin

    # ---------------------------------------------------------
    # Public API: chunk a file
    # ---------------------------------------------------------
    def chunk_file(self, path: str | Path) -> List[Dict[str, Any]]:
        path = Path(path)
        logger.debug(f"{CHUNKING} Chunking file: {path}")

        # -----------------------------
        # Validate file path
        # -----------------------------
        if not path.exists() or not path.is_file():
            logger.error(f"{CHUNKING} File does not exist or is not a file: {path}")
            return []

        # -----------------------------
        # Read file safely
        # -----------------------------
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.error(f"{CHUNKING} Failed reading file '{path}': {e}")
            raise IngestionConfigError(
                f"Failed reading file for chunking: {path}"
            ) from e

        # -----------------------------
        # Build base metadata
        # -----------------------------
        base_meta: Dict[str, Any] = {
            "source_file": str(path),
        }

        # -----------------------------
        # Delegate to plugin
        # -----------------------------
        try:
            chunks = self.plugin.chunk_text(text, base_meta)
        except IngestionChunkingError:
            # Already a domain-specific error, just bubble it up.
            raise
        except Exception as e:
            logger.error(f"{CHUNKING} Chunking plugin failed for '{path}': {e}")
            raise IngestionChunkingError(
                f"Chunking plugin failed for file '{path}'"
            ) from e

        logger.debug(f"{CHUNKING} Extracted {len(chunks)} chunks from '{path}'")

        return chunks
