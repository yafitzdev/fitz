# pipeline/context/steps/pack.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalize import ChunkDict, _to_chunk_dict


@dataclass
class PackWindowStep:
    """
    Pack chunks into a max-character window.

    Rules:
    - Include the first chunk ALWAYS (even if > max_chars)
    - For subsequent chunks: include ONLY if adding them stays <= max_chars
    - No splitting of chunks
    """

    def __call__(self, chunks: list[Any], max_chars: int | None = None) -> list[ChunkDict]:
        canonical = [_to_chunk_dict(ch) for ch in chunks]

        if max_chars is None:
            return canonical

        packed: list[ChunkDict] = []
        total = 0

        for idx, c in enumerate(canonical):
            block_len = len(c.get("content", ""))

            if idx == 0:
                packed.append(c)
                total += block_len
                continue

            if total + block_len <= max_chars:
                packed.append(c)
                total += block_len
            else:
                break

        return packed
