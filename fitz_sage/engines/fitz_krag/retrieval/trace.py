# fitz_sage/engines/fitz_krag/retrieval/trace.py
"""Trace serialization helpers for benchmark-grade retrieval output."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any


def address_trace(address: Any, *, rank: int | None = None) -> dict[str, Any]:
    """Serialize an Address-like object for retrieval traces."""
    payload = {
        "kind": _enum_value(getattr(address, "kind", None)),
        "source_id": getattr(address, "source_id", ""),
        "location": getattr(address, "location", ""),
        "summary": getattr(address, "summary", ""),
        "score": getattr(address, "score", None),
        "metadata": _json_safe(getattr(address, "metadata", {}) or {}),
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def addresses_trace(addresses: list[Any]) -> list[dict[str, Any]]:
    """Serialize an ordered address list with ranks."""
    return [address_trace(address, rank=index) for index, address in enumerate(addresses, start=1)]


def read_result_trace(result: Any, *, rank: int | None = None) -> dict[str, Any]:
    """Serialize a ReadResult-like object for retrieval traces."""
    payload = {
        "address": address_trace(getattr(result, "address", None)),
        "file_path": getattr(result, "file_path", ""),
        "line_range": list(result.line_range) if getattr(result, "line_range", None) else None,
        "content_chars": len(getattr(result, "content", "") or ""),
        "metadata": _json_safe(getattr(result, "metadata", {}) or {}),
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def read_results_trace(results: list[Any]) -> list[dict[str, Any]]:
    """Serialize ordered read results with ranks."""
    return [read_result_trace(result, rank=index) for index, result in enumerate(results, start=1)]


def _enum_value(value: Any) -> Any:
    """Return enum value when present."""
    if isinstance(value, Enum):
        return value.value
    return value


def _json_safe(value: Any) -> Any:
    """Convert common non-JSON values while preserving simple scalars."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


__all__ = ["address_trace", "addresses_trace", "read_result_trace", "read_results_trace"]
