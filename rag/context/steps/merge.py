# rag/context/steps/merge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalize import _to_chunk_dict, ChunkDict


@dataclass
class MergeAdjacentStep:
    """
    Merge chunks that belong to the same document.

    Contract:
    - sorts by chunk_index within each document group
    - merges content with newline separators
    """

    def __call__(self, chunks: list[Any]) -> list[ChunkDict]:
        if not chunks:
            return []

        normed = [_to_chunk_dict(ch) for ch in chunks]
        normed.sort(key=lambda c: int(c.get("chunk_index", 0)))

        merged: list[ChunkDict] = []
        buffer = dict(normed[0])
        buffer_content = buffer.get("content", "")

        for curr in normed[1:]:
            buffer_content += "\n" + curr.get("content", "")

        buffer["content"] = buffer_content
        merged.append(buffer)  # type: ignore[arg-type]

        return merged
