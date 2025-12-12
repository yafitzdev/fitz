from __future__ import annotations

from dataclasses import dataclass
from typing import List, Any

from .normalize import _to_chunk_dict, ChunkDict


@dataclass
class PackWindowStep:
    """
    Pack merged chunks into a max-character window.

    Rules:
    - Include the first chunk ALWAYS (even if > max_chars)
    - For subsequent chunks: include ONLY if adding them stays <= max_chars
    - No splitting of chunks
    """

    def __call__(self, chunks: List[Any], max_chars: int | None = None) -> List[ChunkDict]:
        if max_chars is None:
            # no limit → return all as canonical dicts
            return [_to_chunk_dict(ch) for ch in chunks]

        packed: List[ChunkDict] = []
        total = 0

        for idx, ch in enumerate(chunks):
            c = _to_chunk_dict(ch)
            text = c.get("text", "")
            block_len = len(text)

            if idx == 0:
                # always include first block
                packed.append(c)
                total += block_len
                continue

            # subsequent blocks must fit the limit
            if total + block_len <= max_chars:
                packed.append(c)
                total += block_len
            else:
                # stop packing entirely
                break

        return packed
