# fitz_ai/ingest/chunking/router.py
"""
ChunkingRouter - Routes documents to file-type specific chunkers.

Architecture:
    ┌─────────────────────────────────────┐
    │         ChunkingRouter              │
    │  Routes files to appropriate chunker│
    └─────────────────────────────────────┘
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
   SimpleChunker  MarkdownChunker  PDFChunker
    (.txt, .py)     (.md)          (.pdf)

Usage:
    router = ChunkingRouter.from_config(config)
    chunks = router.chunk_document(raw_doc)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.base import ChunkerPlugin
from fitz_ai.ingest.chunking.registry import get_chunking_plugin
from fitz_ai.engines.classic_rag.config import ChunkingRouterConfig
from fitz_ai.ingest.exceptions.chunking import IngestionChunkingError

if TYPE_CHECKING:
    from fitz_ai.ingest.ingestion.base import RawDocument

logger = logging.getLogger(__name__)


class ChunkingRouter:
    """
    Routes documents to file-type specific chunkers.

    The router maintains instantiated chunker plugins for each configured
    extension, plus a default fallback chunker.
    """

    def __init__(
        self,
        chunker_map: Dict[str, ChunkerPlugin],
        default_chunker: ChunkerPlugin,
        warn_on_fallback: bool = True,
    ) -> None:
        """
        Initialize the router with pre-built chunkers.

        Args:
            chunker_map: Mapping of extensions to chunker instances.
                         Extensions should include the dot (e.g., ".md")
            default_chunker: Fallback chunker for unknown extensions.
            warn_on_fallback: Whether to log a warning when using fallback.
        """
        self._chunker_map = chunker_map
        self._default_chunker = default_chunker
        self._warn_on_fallback = warn_on_fallback
        self._warned_extensions: set[str] = set()

    @classmethod
    def from_config(cls, config: ChunkingRouterConfig) -> "ChunkingRouter":
        """
        Build a router from configuration.

        Args:
            config: Router configuration with default and per-extension settings.

        Returns:
            Configured ChunkingRouter instance.

        Raises:
            IngestionChunkingError: If a plugin cannot be loaded or instantiated.
        """
        default_chunker = cls._build_chunker(
            config.default.plugin_name,
            config.default.kwargs,
            context="default",
        )

        chunker_map: Dict[str, ChunkerPlugin] = {}
        for ext, ext_config in config.by_extension.items():
            normalized_ext = cls._normalize_ext(ext)
            chunker_map[normalized_ext] = cls._build_chunker(
                ext_config.plugin_name,
                ext_config.kwargs,
                context=f"extension '{normalized_ext}'",
            )

        return cls(
            chunker_map=chunker_map,
            default_chunker=default_chunker,
            warn_on_fallback=config.warn_on_fallback,
        )

    @staticmethod
    def _normalize_ext(ext: str) -> str:
        """Normalize extension to lowercase with dot prefix."""
        ext = ext.lower()
        return ext if ext.startswith(".") else f".{ext}"

    @staticmethod
    def _build_chunker(
        plugin_name: str,
        kwargs: Dict[str, Any],
        context: str,
    ) -> ChunkerPlugin:
        """Build a chunker instance from plugin name and kwargs."""
        try:
            PluginCls = get_chunking_plugin(plugin_name)
        except Exception as e:
            raise IngestionChunkingError(
                f"Unknown chunker plugin '{plugin_name}' for {context}: {e}"
            ) from e

        try:
            return PluginCls(**kwargs)
        except Exception as e:
            raise IngestionChunkingError(
                f"Failed to initialize chunker '{plugin_name}' for {context}: {e}"
            ) from e

    def get_chunker(self, ext: str) -> ChunkerPlugin:
        """
        Get the appropriate chunker for a file extension.

        Args:
            ext: File extension (e.g., ".md", ".py", "txt").

        Returns:
            ChunkerPlugin for the extension, or default if not configured.
        """
        normalized = self._normalize_ext(ext)
        chunker = self._chunker_map.get(normalized)

        if chunker is not None:
            return chunker

        # Fallback to default with warning
        if self._warn_on_fallback and normalized not in self._warned_extensions:
            logger.warning(
                f"No chunker configured for extension '{normalized}', "
                f"using default chunker '{self._default_chunker.plugin_name}'"
            )
            self._warned_extensions.add(normalized)

        return self._default_chunker

    def get_chunker_id(self, ext: str) -> str:
        """Get the chunker_id for a file extension."""
        return self.get_chunker(ext).chunker_id

    def chunk_document(
        self,
        raw_doc: "RawDocument",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Route a document to the correct chunker and chunk it.

        Args:
            raw_doc: RawDocument with path, content, and metadata.
            extra_meta: Additional metadata to merge into base_meta.

        Returns:
            List of Chunk objects.

        Raises:
            IngestionChunkingError: If chunking fails.
        """
        path_str = raw_doc.path
        path = Path(path_str)
        ext = path.suffix.lower() or ".txt"

        if not raw_doc.content or not raw_doc.content.strip():
            logger.debug(f"Empty content for document: {path_str}")
            return []

        base_meta: Dict[str, Any] = {
            "source_file": str(path),
            "doc_id": path.stem,
            "file_extension": ext,
            **(raw_doc.metadata or {}),
        }
        if extra_meta:
            base_meta.update(extra_meta)

        chunker = self.get_chunker(ext)
        logger.debug(
            f"Chunking '{path.name}' with {chunker.plugin_name} "
            f"(chunker_id: {chunker.chunker_id})"
        )

        try:
            return chunker.chunk_text(raw_doc.content, base_meta)
        except Exception as e:
            raise IngestionChunkingError(
                f"Chunking failed for '{path_str}' with {chunker.plugin_name}: {e}"
            ) from e

    @property
    def default_chunker(self) -> ChunkerPlugin:
        """Get the default fallback chunker."""
        return self._default_chunker

    @property
    def configured_extensions(self) -> List[str]:
        """Get list of extensions with specific chunker configurations."""
        return sorted(self._chunker_map.keys())

    def __repr__(self) -> str:
        ext_list = ", ".join(self.configured_extensions) or "(none)"
        return (
            f"ChunkingRouter(default={self._default_chunker.plugin_name}, "
            f"extensions=[{ext_list}])"
        )


__all__ = ["ChunkingRouter"]