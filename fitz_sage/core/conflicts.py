# fitz_sage/core/conflicts.py
"""
Conflict detection stub kept for hierarchy-enricher compatibility.

Conflict detection moved to query time in v0.13.0: the pyrrho classifier
(`fitz_sage.governance.pyrrho`) decides TRUSTWORTHY / DISPUTED / ABSTAIN
in a single forward pass over (query, contexts). No ingest-time conflict
extraction is needed.
"""

from __future__ import annotations

from typing import Sequence

from fitz_sage.core.chunk import Chunk


def find_conflicts(chunks: Sequence[Chunk]) -> list[tuple[str, str, str, str]]:
    """Ingest-time conflict-detection stub. Returns an empty list.

    Kept so the hierarchy enricher can call it unconditionally;
    actual conflict detection happens at query time via pyrrho.
    """
    return []


__all__ = ["find_conflicts"]
