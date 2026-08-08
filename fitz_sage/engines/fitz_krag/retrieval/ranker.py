# fitz_sage/engines/fitz_krag/retrieval/ranker.py
"""
Cross-strategy ranker — scores and ranks addresses from multiple strategies.

Uses strategy weights, entity matching bonuses, and type mismatch
penalties to produce a coherent ranking across code and document results.
"""

from __future__ import annotations

import logging
from typing import Any

from fitz_sage.engines.fitz_krag.types import Address, AddressKind

logger = logging.getLogger(__name__)

# Bonus for entity name match in address summary/location
ENTITY_MATCH_BONUS = 0.15

# Strategy kind mapping
_KIND_TO_STRATEGY: dict[AddressKind, str] = {
    AddressKind.SYMBOL: "code",
    AddressKind.FILE: "code",
    AddressKind.SECTION: "section",
    AddressKind.TABLE: "table",
}


class CrossStrategyRanker:
    """Ranks addresses across strategies using retrieval profile."""

    def rank(
        self,
        addresses: list[Address],
        profile: Any = None,
    ) -> list[Address]:
        """
        Score and rank addresses using strategy weights from profile.

        Scoring:
        1. Base score from strategy's internal ranking (addr.score)
        2. Strategy weight multiplier from profile
        3. Entity match bonus when query mentions symbol/section name
        """
        weights = profile.strategy_weights if profile else None
        entities = set(e.lower() for e in profile.entities) if profile else set()

        scored: list[tuple[float, Address]] = []
        for addr in addresses:
            score = self._compute_score(addr, weights, entities)
            scored.append((score, addr))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [addr for _, addr in scored]

    def _compute_score(
        self,
        addr: Address,
        weights: dict[str, float] | None,
        entities: set[str],
    ) -> float:
        """Compute weighted score for an address."""
        # Base score from strategy
        base_score = addr.score

        # Strategy weight multiplier
        if weights:
            strategy = _KIND_TO_STRATEGY[addr.kind]
            weight = weights.get(strategy, 0.1)
            weighted_score = base_score * weight
        else:
            weighted_score = base_score

        # Entity match bonus
        if entities:
            location_lower = addr.location.lower()
            summary_lower = (addr.summary or "").lower()
            for entity in entities:
                if entity in location_lower or entity in summary_lower:
                    weighted_score += ENTITY_MATCH_BONUS
                    break

        return weighted_score
