# pipeline/context/steps/normalize.py
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
    """
    if isinstance(chunk_like, dict):
        data = dict(chunk_like)
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        doc_id = str(
            data.get("doc_id") or data.get("document_id") or data.get("source") or "unknown"
        )
        chunk_index = int(data.get("chunk_index") if data.get("chunk_index") is not None else 0)
        chunk_id = data.get("id")
        if chunk_id is None or str(chunk_id).strip() == "":
            chunk_id = f"{doc_id}:{chunk_index}"

        return {
            "id": str(chunk_id),
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "content": str(data.get("content") or ""),
            "metadata": dict(metadata),
        }

    metadata = getattr(chunk_like, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    doc_id = str(
        getattr(chunk_like, "doc_id", None)
        or getattr(chunk_like, "document_id", None)
        or getattr(chunk_like, "source", None)
        or "unknown"
    )
    chunk_index = int(getattr(chunk_like, "chunk_index", 0) or 0)

    chunk_id = getattr(chunk_like, "id", None)
    if chunk_id is None or str(chunk_id).strip() == "":
        chunk_id = f"{doc_id}:{chunk_index}"

    return {
        "id": str(chunk_id),
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "content": str(getattr(chunk_like, "content", "") or ""),
        "metadata": dict(metadata),
    }


@dataclass
class NormalizeStep:
    """
    Canonicalize incoming chunks into `ChunkDict` form.

    This step is the single point where we accept "chunk-like" inputs.
    All subsequent steps should operate on `ChunkDict`.
    """

    def __call__(self, chunks: list[Any]) -> list[ChunkDict]:
        return [_to_chunk_dict(ch) for ch in chunks]
