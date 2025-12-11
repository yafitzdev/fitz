from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Dict

from .normalize import _normalize_text, _to_chunk_dict, ChunkDict


@dataclass
class DedupeStep:
    """
    Deduplicate chunks based on normalized text.
    The first occurrence is kept.

    Returns canonical dict chunks.
    """

    def __call__(self, chunks: List[Any]) -> List[ChunkDict]:
        seen = set()
        output: List[ChunkDict] = []

        for ch in chunks:
            c = _to_chunk_dict(ch)
            text_norm = _normalize_text(c["text"])
            if text_norm in seen:
                continue

            seen.add(text_norm)

            c_out = dict(c)
            c_out["text"] = text_norm
            output.append(c_out)

        return output
