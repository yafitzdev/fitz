# fitz_ai/engines/classic_rag/pipeline/steps/normalize.py
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


def _get_attr(obj: Any, *keys: str, default: Any = None) -> Any:
    """
    Get attribute from dict or object, trying multiple keys.

    Args:
        obj: Dict or object to extract from
        *keys: Keys/attributes to try in order
        default: Default value if none found
    """
    is_dict = isinstance(obj, dict)
    for key in keys:
        val = obj.get(key) if is_dict else getattr(obj, key, None)
        if val is not None:
            return val
    return default


def _to_chunk_dict(chunk_like: Any) -> ChunkDict:
    """
    Convert chunk-like objects (dict or dataclass) into canonical dict form.
    """
    metadata = _get_attr(chunk_like, "metadata", default={})
    if not isinstance(metadata, Mapping):
        metadata = {}

    doc_id = str(
        _get_attr(chunk_like, "doc_id", "document_id", "source", default="unknown")
    )

    chunk_index_raw = _get_attr(chunk_like, "chunk_index", default=0)
    chunk_index = int(chunk_index_raw) if chunk_index_raw is not None else 0

    chunk_id = _get_attr(chunk_like, "id")
    if chunk_id is None or str(chunk_id).strip() == "":
        chunk_id = f"{doc_id}:{chunk_index}"

    content = str(_get_attr(chunk_like, "content", default="") or "")

    return {
        "id": str(chunk_id),
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "content": content,
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
