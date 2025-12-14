# pipeline/context/steps/group.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalize import ChunkDict, _to_chunk_dict


@dataclass
class GroupByDocumentStep:
    """
    Group chunks by their document id.

    Priority:
    - top-level "doc_id"
    - attribute .doc_id
    - metadata["doc_id"]
    - metadata["file"]
    - "unknown"

    Returns:
        dict[str, list[ChunkDict]]
    """

    def __call__(self, chunks: list[Any]) -> dict[str, list[ChunkDict]]:
        groups: dict[str, list[ChunkDict]] = {}

        for ch in chunks:
            c = _to_chunk_dict(ch)

            doc_id = c.get("doc_id") or ""
            if not doc_id:
                meta = c.get("metadata", {}) or {}
                doc_id = str(meta.get("doc_id") or meta.get("file") or "unknown")
                c["doc_id"] = doc_id

            groups.setdefault(str(doc_id), []).append(c)

        return groups
