# fitz_ai/ingest/enrichment/hierarchy/matcher.py
"""
Chunk path matching for hierarchy rules.

Filters chunks based on glob patterns matching their file_path metadata.
Uses fnmatch for Unix shell-style pattern matching.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk

logger = logging.getLogger(__name__)


class ChunkMatcher:
    """
    Matches chunks against glob patterns.

    Uses fnmatch for glob-style pattern matching on chunk file paths.
    Paths are normalized to POSIX style for consistent matching across platforms.

    Example:
        >>> matcher = ChunkMatcher(["comments/**", "feedback/*.md"])
        >>> matcher.matches(chunk_with_path_comments_video1_txt)
        True
    """

    def __init__(self, patterns: List[str]):
        """
        Initialize matcher with glob patterns.

        Args:
            patterns: List of glob patterns (e.g., ["comments/**", "*.txt"])
        """
        self._patterns = patterns

    def matches(self, chunk: "Chunk") -> bool:
        """
        Check if a chunk's file path matches any pattern.

        Args:
            chunk: Chunk to check

        Returns:
            True if file_path matches any pattern
        """
        file_path = chunk.metadata.get("file_path") or chunk.metadata.get("source_file")
        if not file_path:
            return False

        # Normalize to POSIX-style relative path
        normalized = self._normalize_path(str(file_path))

        return any(fnmatch.fnmatch(normalized, pattern) for pattern in self._patterns)

    def filter_chunks(self, chunks: List["Chunk"]) -> List["Chunk"]:
        """
        Filter chunks that match any pattern.

        Args:
            chunks: List of chunks to filter

        Returns:
            Chunks whose file_path matches at least one pattern
        """
        return [c for c in chunks if self.matches(c)]

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path to POSIX style for consistent matching.

        Converts Windows backslashes to forward slashes and
        extracts just the relative-looking portion.
        """
        # Handle Windows paths
        posix_path = path.replace("\\", "/")

        # If it's an absolute path, try to extract a relative-like portion
        # This handles cases like "C:/Users/foo/project/comments/video.md"
        # where we want to match against "comments/**"
        if "/" in posix_path:
            parts = posix_path.split("/")
            # Return the path as-is but normalized
            return "/".join(parts)

        return posix_path


__all__ = ["ChunkMatcher"]
