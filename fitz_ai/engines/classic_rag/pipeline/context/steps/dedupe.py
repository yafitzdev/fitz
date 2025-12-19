# pipeline/context/steps/dedupe.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalize import ChunkDict, _normalize_text, _to_chunk_dict


@dataclass
class DedupeStep:
    """
    Deduplicate chunks based on normalized content.

    The first occurrence is kept.

    Returns canonical dict chunks.
    """

    def __call__(self, chunks: list[Any]) -> list[ChunkDict]:
        seen: set[str] = set()
        output: list[ChunkDict] = []

        for ch in chunks:
            c = _to_chunk_dict(ch)
            key = _normalize_text(c["content"])
            if key in seen:
                continue

            seen.add(key)
            output.append(c)

        return output
