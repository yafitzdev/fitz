from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .normalize import _to_chunk_dict, ChunkDict


@dataclass
class PackWindowStep:
    """
    Return the largest prefix of chunks whose total text length
    does not exceed `max_chars`.

    Input:
        - List of chunk-like objects (dicts or legacy objects)

    Output:
        - List[ChunkDict] in canonical form
    """

    max_chars: int = 6000

    def __call__(self, chunks: List[Any], max_chars: int | None = None) -> List[ChunkDict]:
        limit = max_chars if max_chars is not None else self.max_chars

        packed: List[ChunkDict] = []
        total = 0

        for ch in chunks:
            c = _to_chunk_dict(ch)
            t = c.get("text", "")
            if not isinstance(t, str):
                t = str(t)

            if total + len(t) > limit:
                break

            packed.append(c)
            total += len(t)

        return packed
