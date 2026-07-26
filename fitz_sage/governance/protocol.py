# fitz_sage/governance/protocol.py
"""
EvidenceItem Protocol - Generic interface for evidence that constraints can evaluate.

KRAG's ReadResult satisfies this protocol without adapter code. Custom engines
can provide any object with the same evidence fields.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EvidenceItem(Protocol):
    """Any retrieved evidence that constraints can evaluate."""

    content: str
    metadata: dict[str, Any]


__all__ = ["EvidenceItem"]
