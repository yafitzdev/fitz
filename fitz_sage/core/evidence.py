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
