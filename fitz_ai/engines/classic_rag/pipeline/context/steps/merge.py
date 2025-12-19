# pipeline/context/steps/merge.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalize import ChunkDict, _to_chunk_dict


@dataclass
class MergeAdjacentStep:
    """
    Merge only truly adjacent chunks (consecutive chunk_index) within a document.

    Contract:
    - input is expected to be chunks from ONE document (already grouped)
    - sorts by chunk_index
    - merges runs where chunk_index increments by 1
    - preserves the first chunk's id/chunk_index and records merged ids in metadata
    """

    def __call__(self, chunks: list[Any]) -> list[ChunkDict]:
        if not chunks:
            return []

        normed = [_to_chunk_dict(ch) for ch in chunks]
        normed.sort(key=lambda c: int(c.get("chunk_index", 0)))

        merged: list[ChunkDict] = []

        current = dict(normed[0])
        current_meta = dict(current.get("metadata") or {})
        merged_ids = [current["id"]]
        prev_index = int(current.get("chunk_index", 0))

        for ch in normed[1:]:
            chd = dict(ch)
            ch_index = int(chd.get("chunk_index", 0))

            if ch_index == prev_index + 1:
                current["content"] = f"{current.get('content', '')}\n{chd.get('content', '')}"
                merged_ids.append(chd["id"])
                prev_index = ch_index
                continue

            if len(merged_ids) > 1:
                current_meta = dict(current.get("metadata") or {})
                current_meta["merged_from_ids"] = list(merged_ids)
                current["metadata"] = current_meta

            merged.append(current)  # type: ignore[arg-type]

            current = chd
            current_meta = dict(current.get("metadata") or {})
            merged_ids = [current["id"]]
            prev_index = ch_index

        if len(merged_ids) > 1:
            current_meta = dict(current.get("metadata") or {})
            current_meta["merged_from_ids"] = list(merged_ids)
            current["metadata"] = current_meta

        merged.append(current)  # type: ignore[arg-type]
        return merged
