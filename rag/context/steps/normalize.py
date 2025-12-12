# rag/context/steps/normalize.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypedDict


class ChunkDict(TypedDict):
    id: str
    doc_id: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]


def _normalize_text(text: str) -> str:
    """
    Normalize text for deduplication keys:
    - strip leading/trailing whitespace
    - collapse internal whitespace to single spaces
    """
    return " ".join(str(text or "").split())


def _to_chunk_dict(chunk_like: Any) -> ChunkDict:
    """
    Convert chunk-like objects into canonical dict form.

    Accepts:
    - rag.models.chunk.Chunk (or any object with .id/.doc_id/.chunk_index/.content/.metadata)
    - dicts with keys: id, doc_id, chunk_index, content, metadata

    Returns:
        {
            "id": str,
            "doc_id": str,
            "chunk_index": int,
            "content": str,
            "metadata": dict,
        }
    """
    if isinstance(chunk_like, dict):
        data = dict(chunk_like)

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        return {
            "id": str(data.get("id") or ""),
            "doc_id": str(data.get("doc_id") or ""),
            "chunk_index": int(data.get("chunk_index") or 0),
            "content": str(data.get("content") or ""),
            "metadata": dict(metadata),
        }

    metadata = getattr(chunk_like, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    return {
        "id": str(getattr(chunk_like, "id", "") or ""),
        "doc_id": str(getattr(chunk_like, "doc_id", "") or ""),
        "chunk_index": int(getattr(chunk_like, "chunk_index", 0) or 0),
        "content": str(getattr(chunk_like, "content", "") or ""),
        "metadata": dict(metadata),
    }


@dataclass
class NormalizeStep:
    """
    Normalization step for context pipeline.

    Currently a no-op step kept for pipeline stability.
    """

    def __call__(self, chunks: list[Any]) -> list[Any]:
        return chunks
