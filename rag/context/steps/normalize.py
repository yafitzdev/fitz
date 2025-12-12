from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

ChunkDict = Dict[str, Any]


def _normalize_text(text: str) -> str:
    """
    Normalize text for deduplication:
    - strip leading/trailing whitespace
    - collapse internal whitespace to single spaces
    """
    if text is None:
        return ""
    return " ".join(str(text).split())


def _to_chunk_dict(chunk_like: Any) -> ChunkDict:
    """
    Universal normalization layer for 'chunks'.

    Accepts:
    - dicts with keys like text, metadata, id, score, file
    - legacy Chunk objects with .text/.content/.metadata/.id/.score
    - simple test objects with .content and .metadata

    Returns a canonical dict:

        {
            "id": str | None,
            "text": str,
            "metadata": dict,
            "score": float | None,
        }

    Also normalizes 'file' from either metadata["file"] or top-level "file".
    """
    # If it's already a dict, start from a shallow copy
    if isinstance(chunk_like, dict):
        data = dict(chunk_like)
        text = (
            data.get("text")
            if data.get("text") is not None
            else data.get("content")
        )
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}

        # If there's a top-level 'file', ensure it is reflected in metadata
        file_val = data.get("file") or metadata.get("file")
        if file_val is not None:
            metadata = dict(metadata)
            metadata.setdefault("file", file_val)

        return {
            "id": data.get("id"),
            "text": str(text) if text is not None else "",
            "metadata": metadata,
            "score": data.get("score"),
        }

    # Fallback: treat as an object with attributes
    text = getattr(chunk_like, "text", None)
    if text is None:
        text = getattr(chunk_like, "content", "")

    metadata = getattr(chunk_like, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    cid = getattr(chunk_like, "id", None)
    score = getattr(chunk_like, "score", None)

    # Optional 'file' attribute
    file_val = getattr(chunk_like, "file", None) or metadata.get("file")
    if file_val is not None:
        metadata = dict(metadata)
        metadata.setdefault("file", file_val)

    return {
        "id": cid,
        "text": str(text),
        "metadata": metadata,
        "score": score,
    }


@dataclass
class NormalizeStep:
    """
    Normalization step for context pipeline.

    Right now this is effectively a no-op that just returns the
    list as-is. Having it as a dedicated step allows us to add
    per-chunk normalization later without changing pipeline wiring.
    """

    def __call__(self, chunks: list[Any]) -> list[Any]:
        # Future hook: apply per-chunk normalization if needed.
        # For now, keep behavior identical to the existing system.
        return chunks
