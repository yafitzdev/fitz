# fitz_sage/core/evidence.py
"""Serializable evidence contracts for retrieval-first workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .answer_mode import AnswerMode


@dataclass
class EvidenceItem:
    """One ranked source unit in an evidence pack."""

    rank: int
    source_id: str
    file_path: str
    address_kind: str
    address_location: str
    line_range: tuple[int, int] | None
    score: float | None
    excerpt: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "rank": self.rank,
            "source_id": self.source_id,
            "file_path": self.file_path,
            "address_kind": self.address_kind,
            "address_location": self.address_location,
            "line_range": list(self.line_range) if self.line_range else None,
            "score": self.score,
            "excerpt": self.excerpt,
            "content": self.content,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvidenceItem":
        """Build an evidence item from its serialized representation."""
        line_range = raw.get("line_range")
        parsed_range = None
        if isinstance(line_range, (list, tuple)) and len(line_range) == 2:
            parsed_range = (int(line_range[0]), int(line_range[1]))
        metadata = raw.get("metadata")
        score = raw.get("score")
        return cls(
            rank=int(raw.get("rank", 0)),
            source_id=str(raw.get("source_id") or ""),
            file_path=str(raw.get("file_path") or ""),
            address_kind=str(raw.get("address_kind") or ""),
            address_location=str(raw.get("address_location") or ""),
            line_range=parsed_range,
            score=float(score) if score is not None else None,
            excerpt=str(raw.get("excerpt") or ""),
            content=str(raw.get("content") or ""),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


@dataclass
class EvidencePack:
    """Ranked, governed evidence for a query."""

    query: str
    mode: AnswerMode | None
    items: list[EvidenceItem] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    indexing_status: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dictionary."""
        return {
            "query": self.query,
            "mode": self.mode.value if self.mode is not None else None,
            "items": [item.to_dict() for item in self.items],
            "reasons": list(self.reasons),
            "timings": _json_safe(self.timings),
            "indexing_status": _json_safe(self.indexing_status),
            "metadata": _json_safe(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the evidence pack to JSON."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvidencePack":
        """Build an evidence pack from its serialized representation."""
        raw_mode = raw.get("mode")
        mode = AnswerMode(str(raw_mode)) if raw_mode is not None else None
        items = raw.get("items")
        timings = raw.get("timings")
        indexing_status = raw.get("indexing_status")
        metadata = raw.get("metadata")
        reasons = raw.get("reasons")
        return cls(
            query=str(raw.get("query") or ""),
            mode=mode,
            items=[
                EvidenceItem.from_dict(item)
                for item in (items if isinstance(items, list) else [])
                if isinstance(item, dict)
            ],
            reasons=[
                str(reason)
                for reason in (reasons if isinstance(reasons, list) else [])
                if isinstance(reason, str)
            ],
            timings=dict(timings) if isinstance(timings, dict) else {},
            indexing_status=(dict(indexing_status) if isinstance(indexing_status, dict) else {}),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvidencePack":
        """Build an evidence pack from JSON."""
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("EvidencePack JSON must contain an object.")
        return cls.from_dict(raw)


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON values while preserving simple scalars."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["EvidenceItem", "EvidencePack"]
