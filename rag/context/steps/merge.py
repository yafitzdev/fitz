from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .normalize import _to_chunk_dict, ChunkDict


@dataclass
class MergeAdjacentStep:
    """
    Merge adjacent chunks that belong to the same document (same 'file').

    Input:
        - List of chunk-like objects (dicts or legacy objects)

    Output:
        - List[ChunkDict] in canonical form, with merged text per document-run.
    """

    def __call__(self, chunks: List[Any]) -> List[ChunkDict]:
        if not chunks:
            return []

        normed = [_to_chunk_dict(ch) for ch in chunks]
        merged: List[ChunkDict] = []

        def file_of(c: ChunkDict) -> str:
            meta = c.get("metadata", {}) or {}
            return str(meta.get("file", "unknown"))

        buffer = dict(normed[0])
        buffer_text = buffer.get("text", "")
        buffer_file = file_of(buffer)

        for curr in normed[1:]:
            curr_file = file_of(curr)

            if curr_file == buffer_file:
                # same doc → append text
                buffer_text += "\n" + curr.get("text", "")
            else:
                # flush buffer
                buffer["text"] = buffer_text
                merged.append(buffer)

                # start new buffer
                buffer = dict(curr)
                buffer_text = buffer.get("text", "")
                buffer_file = curr_file

        # flush last buffer
        buffer["text"] = buffer_text
        merged.append(buffer)

        return merged
