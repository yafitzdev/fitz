# fitz_ai/ingest/chunking/engine.py
"""
ChunkingEngine - Central controller for chunking operations.

Uses ChunkingRouter to route files to type-specific chunkers based on extension.

Architecture:
    ChunkingEngine
        └── ChunkingRouter
            ├── .md → MarkdownChunker
            ├── .py → PythonChunker
            └── default → SimpleChunker
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.router import ChunkingRouter
from fitz_ai.engines.classic_rag.config import ChunkingRouterConfig
from fitz_ai.ingest.exceptions.chunking import IngestionChunkingError
from fitz_ai.ingest.exceptions.config import IngestionConfigError
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import CHUNKING

logger = get_logger(__name__)


class ChunkingEngine:
    """
    Central controller for chunking operations.

    Routes files to type-specific chunkers via ChunkingRouter.

    Usage:
        engine = ChunkingEngine.from_config(router_config)
        chunks = engine.run(raw_doc)
    """

    def __init__(self, router: ChunkingRouter) -> None:
        """
        Initialize the engine with a router.

        Args:
            router: ChunkingRouter for file-type specific chunking.
        """
        self._router = router

    @classmethod
    def from_config(cls, config: ChunkingRouterConfig) -> "ChunkingEngine":
        """
        Create engine from a ChunkingRouterConfig.

        Args:
            config: Router configuration with default and per-extension settings.

        Returns:
            Configured ChunkingEngine instance.

        Raises:
            IngestionConfigError: If config is invalid.
        """
        try:
            router = ChunkingRouter.from_config(config)
        except IngestionChunkingError as e:
            raise IngestionConfigError(f"Failed to build chunking router: {e}") from e

        return cls(router=router)

    @property
    def router(self) -> ChunkingRouter:
        """Get the router."""
        return self._router

    def get_chunker_id(self, ext: str) -> str:
        """
        Get the chunker_id for a file extension.

        Args:
            ext: File extension.

        Returns:
            The chunker_id string.
        """
        return self._router.get_chunker_id(ext)

    def run(self, raw_doc: object) -> List[Chunk]:
        """
        Chunk a raw document into smaller pieces.

        Automatically selects the chunker based on file extension.

        Args:
            raw_doc: A RawDocument object with 'path', 'content', and 'metadata'.

        Returns:
            List of Chunk objects.

        Raises:
            IngestionChunkingError: If chunking fails.
        """
        path_str = getattr(raw_doc, "path", "unknown")
        path = Path(path_str) if path_str != "unknown" else None

        logger.debug(f"{CHUNKING} Chunking document: {path_str}")

        content = getattr(raw_doc, "content", None)
        if content is None:
            raise IngestionChunkingError(
                f"RawDocument has no 'content' attribute: {path_str}"
            )

        if not content or not content.strip():
            logger.warning(f"{CHUNKING} Empty content for document: {path_str}")
            return []

        # Build base metadata
        base_meta: Dict[str, Any] = {
            "source_file": str(path) if path else path_str,
            "doc_id": path.stem if path else "unknown",
            **(getattr(raw_doc, "metadata", None) or {}),
        }

        ext = path.suffix.lower() if path else ".txt"
        chunker = self._router.get_chunker(ext)

        logger.debug(
            f"{CHUNKING} Using chunker '{chunker.plugin_name}' "
            f"(id: {chunker.chunker_id}) for '{path_str}'"
        )

        try:
            chunks = chunker.chunk_text(content, base_meta)
        except IngestionChunkingError:
            raise
        except Exception as e:
            logger.error(f"{CHUNKING} Chunking plugin failed for '{path_str}': {e}")
            raise IngestionChunkingError(
                f"Chunking plugin failed for file '{path_str}'"
            ) from e

        logger.debug(f"{CHUNKING} Extracted {len(chunks)} chunks from '{path_str}'")
        return chunks


__all__ = ["ChunkingEngine"]