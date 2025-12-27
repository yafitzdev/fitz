# fitz_ai/ingest/enrichment/hierarchy/grouper.py
"""
Chunk grouping by metadata keys.

Groups filtered chunks by a specified metadata key for hierarchical summarization.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk

logger = logging.getLogger(__name__)


class ChunkGrouper:
    """
    Groups chunks by a metadata key.

    Handles missing keys gracefully by placing those chunks in an "_ungrouped" bucket.
    The _ungrouped bucket is skipped during summarization but original chunks are kept.

    Example:
        >>> grouper = ChunkGrouper("video_id")
        >>> groups = grouper.group(chunks)
        >>> groups
        {"abc123": [chunk1, chunk2], "def456": [chunk3], "_ungrouped": [chunk4]}
    """

    def __init__(self, group_by: str):
        """
        Initialize grouper.

        Args:
            group_by: Metadata key to group chunks by
        """
        self._group_by = group_by

    def group(self, chunks: List["Chunk"]) -> Dict[str, List["Chunk"]]:
        """
        Group chunks by the configured metadata key.

        Args:
            chunks: Chunks to group

        Returns:
            Dict mapping group_key -> list of chunks.
            Chunks missing the key go to "_ungrouped".
        """
        groups: Dict[str, List["Chunk"]] = defaultdict(list)
        missing_count = 0

        for chunk in chunks:
            group_key = chunk.metadata.get(self._group_by)

            if group_key is None:
                groups["_ungrouped"].append(chunk)
                missing_count += 1
            else:
                groups[str(group_key)].append(chunk)

        if missing_count > 0:
            logger.warning(
                f"[HIERARCHY] {missing_count} chunks missing '{self._group_by}' key, "
                f"placed in '_ungrouped' group"
            )

        logger.info(
            f"[HIERARCHY] Grouped {len(chunks)} chunks into {len(groups)} groups "
            f"by '{self._group_by}'"
        )

        return dict(groups)


__all__ = ["ChunkGrouper"]
