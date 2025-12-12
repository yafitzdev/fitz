from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .normalize import _to_chunk_dict, ChunkDict


@dataclass
class GroupByDocumentStep:
    """
    Group chunks by their 'file' origin.

    Priority:
    - metadata["file"]
    - top-level "file"
    - "unknown"

    Returns:
        Dict[str, List[ChunkDict]]
        where the key is the file identifier.
    """

    def __call__(self, chunks: List[Any]) -> Dict[str, List[ChunkDict]]:
        groups: Dict[str, List[ChunkDict]] = {}

        for ch in chunks:
            c = _to_chunk_dict(ch)
            meta = c.get("metadata", {}) or {}
            file_val = meta.get("file") or getattr(ch, "file", None)

            if file_val is None and isinstance(ch, dict):
                file_val = ch.get("file")

            if file_val is None:
                file_val = "unknown"

            # Ensure metadata["file"] is set for downstream consumers
            meta = dict(meta)
            meta.setdefault("file", file_val)
            c["metadata"] = meta

            groups.setdefault(str(file_val), []).append(c)

        return groups
