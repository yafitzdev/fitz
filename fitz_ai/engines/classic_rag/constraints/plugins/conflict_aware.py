# fitz_ai/engines/classic_rag/constraints/plugins/conflict_aware.py
"""
Conflict-Aware Constraint - Default guardrail for contradiction detection.

This constraint detects explicitly conflicting claims in retrieved chunks.
When conflicts exist, it prevents the system from giving a confident answer.

Detects conflicts in:
- Classifications (security vs operational incident)
- Trends (improved vs declined)
- Sentiment (positive vs negative)
- State (successful vs failed, approved vs rejected)
- Numeric claims (significantly different values)

This constraint does NOT:
- Resolve conflicts
- Choose sides
- Apply authority hierarchies

It only prevents confident collapse when evidence disagrees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE

from ..base import ConstraintResult

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk

logger = get_logger(__name__)


# =============================================================================
# Conflict Detection Patterns
# =============================================================================

# Patterns indicating explicit classification/categorization
CLASSIFICATION_PATTERNS: tuple[re.Pattern, ...] = (
    # Incident/event classification
    re.compile(
        r"\b(?:is|was|are|were)\s+(?:a|an|the)?\s*(\w+(?:\s+\w+)?)\s+(?:incident|event|issue|problem)",
        re.I,
    ),
    re.compile(r"\bclassified\s+as\s+(?:a|an)?\s*['\"]?(\w+(?:\s+\w+)?)['\"]?", re.I),
    re.compile(r"\bcategor(?:y|ized)\s+(?:as|:)\s*['\"]?(\w+(?:\s+\w+)?)['\"]?", re.I),
    re.compile(r"\btype\s*:\s*['\"]?(\w+(?:\s+\w+)?)['\"]?", re.I),
    re.compile(
        r"\b(?:security|operational|infrastructure|network)\s+(?:incident|event|issue)",
        re.I,
    ),
    # Trend indicators
    re.compile(r"\b(improved|increased|grew|rose|gained|surged)\b", re.I),
    re.compile(r"\b(declined|decreased|dropped|fell|lost|plummeted)\b", re.I),
    re.compile(r"\b(stable|unchanged|flat|steady|consistent)\b", re.I),
    # Sentiment indicators
    re.compile(r"\b(?:was|is|were|are)\s+(positive|negative|neutral)\b", re.I),
    re.compile(r"\b(?:response|feedback|sentiment)\s+(?:was|is)\s+(positive|negative|neutral)\b", re.I),
    re.compile(r"\b(good|bad|excellent|poor|great|terrible)\s+(?:results?|performance|outcome)\b", re.I),
    # State indicators (both "was successful" and standalone "failed")
    re.compile(r"\b(?:was|is|were|are)\s+(successful|failed|completed|pending)\b", re.I),
    re.compile(r"\b(successful(?:ly)?|failed|completed|pending)\b", re.I),
    re.compile(r"\b(?:was|is|were|are)\s+(approved|rejected|accepted|denied)\b", re.I),
    # Numeric claims (NPS, scores, percentages)
    re.compile(r"\b(?:nps|score|rating)\s*(?:is|was|of|:)?\s*(\d+(?:\.\d+)?)\b", re.I),
    re.compile(r"\b(\d+(?:\.\d+)?)\s*%\s*(?:increase|decrease|growth|decline)?\b", re.I),
)

# Mutually exclusive pairs (normalized to lowercase)
# Format: (term_a, term_b) - if one chunk has term_a and another has term_b, it's a conflict
CONFLICTING_PAIRS: tuple[tuple[str, str], ...] = (
    # Classification conflicts
    ("security", "operational"),
    ("security incident", "operational incident"),
    ("internal", "external"),
    # State conflicts
    ("confirmed", "unconfirmed"),
    ("approved", "rejected"),
    ("accepted", "denied"),
    ("active", "deprecated"),
    ("enabled", "disabled"),
    ("primary", "secondary"),
    ("successful", "failed"),
    ("completed", "pending"),
    ("yes", "no"),
    ("true", "false"),
    # Trend conflicts
    ("improved", "declined"),
    ("improved", "decreased"),
    ("improved", "dropped"),
    ("improved", "fell"),
    ("increased", "decreased"),
    ("increased", "declined"),
    ("increased", "dropped"),
    ("increased", "fell"),
    ("grew", "dropped"),
    ("grew", "declined"),
    ("grew", "decreased"),
    ("rose", "fell"),
    ("rose", "dropped"),
    ("rose", "declined"),
    ("gained", "lost"),
    ("surged", "plummeted"),
    ("stable", "improved"),
    ("stable", "declined"),
    ("stable", "increased"),
    ("stable", "decreased"),
    ("unchanged", "improved"),
    ("unchanged", "declined"),
    # Sentiment conflicts
    ("positive", "negative"),
    ("good", "bad"),
    ("good", "poor"),
    ("excellent", "poor"),
    ("excellent", "terrible"),
    ("great", "terrible"),
)


def _extract_classifications(text: str) -> set[str]:
    """Extract classification terms from text."""
    classifications: set[str] = set()

    for pattern in CLASSIFICATION_PATTERNS:
        for match in pattern.finditer(text):
            # Get the full match or first group
            term = match.group(1) if match.lastindex else match.group(0)
            if term:
                classifications.add(term.lower().strip())

    return classifications


def _find_conflicts(chunks: Sequence["Chunk"]) -> list[tuple[str, str, str, str]]:
    """
    Find conflicting claims across chunks.

    Detects conflicts in classifications, trends, sentiment, and state.

    Returns list of (chunk1_id, claim1, chunk2_id, claim2) tuples.
    """
    conflicts: list[tuple[str, str, str, str]] = []

    # Extract claims per chunk
    chunk_claims: list[tuple[str, set[str]]] = []
    for chunk in chunks:
        claims = _extract_classifications(chunk.content)
        if claims:
            chunk_claims.append((chunk.id, claims))

    # Compare all pairs
    for i, (id1, claims1) in enumerate(chunk_claims):
        for id2, claims2 in chunk_claims[i + 1 :]:
            for term1 in claims1:
                for term2 in claims2:
                    if _are_conflicting(term1, term2):
                        conflicts.append((id1, term1, id2, term2))

    return conflicts


def _are_conflicting(term1: str, term2: str) -> bool:
    """Check if two terms are mutually exclusive."""
    t1, t2 = term1.lower(), term2.lower()

    # Same term is not a conflict
    if t1 == t2:
        return False

    # Check explicit conflicting pairs
    for a, b in CONFLICTING_PAIRS:
        if (a in t1 and b in t2) or (b in t1 and a in t2):
            return True

    # Check numeric conflicts (same metric type, significantly different values)
    num1 = re.search(r"(\d+(?:\.\d+)?)", t1)
    num2 = re.search(r"(\d+(?:\.\d+)?)", t2)
    if num1 and num2:
        try:
            val1, val2 = float(num1.group(1)), float(num2.group(1))
            # Significant difference (>20% relative difference)
            if val1 > 0 and val2 > 0:
                ratio = max(val1, val2) / min(val1, val2)
                if ratio > 1.2:
                    return True
        except ValueError:
            pass

    return False


def _is_resolution_query(query: str) -> bool:
    """
    Detect if query explicitly asks for conflict resolution.

    Queries like "Which classification should be considered authoritative?"
    should be allowed to give decisive answers even with conflicts.
    """
    resolution_patterns = (
        r"\bwhich\b.*\bauthoritative\b",
        r"\bwhich\b.*\bcorrect\b",
        r"\bwhich\b.*\btrust\b",
        r"\bwhich\b.*\bbelieve\b",
        r"\bresolve\b.*\bconflict",
        r"\breconcile\b",
        r"\bwhy\s+(?:do|does|are)\b.*\bdisagree\b",
        r"\bwhy\s+(?:the)?\s*difference\b",
    )

    query_lower = query.lower()
    for pattern in resolution_patterns:
        if re.search(pattern, query_lower):
            return True

    return False


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class ConflictAwareConstraint:
    """
    Default constraint that detects conflicting classifications.

    When retrieved chunks contain mutually exclusive claims (e.g., one says
    "security incident" and another says "operational incident"), this
    constraint prevents the system from confidently asserting either.

    This does NOT resolve conflicts - it only prevents confident collapse.

    Attributes:
        enabled: Whether this constraint is active (default: True)
    """

    enabled: bool = True

    @property
    def name(self) -> str:
        return "conflict_aware"

    def apply(
        self,
        query: str,
        chunks: Sequence["Chunk"],
    ) -> ConstraintResult:
        """
        Check for conflicting classifications in retrieved chunks.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies decisive answer if conflicts detected
        """
        if not self.enabled:
            return ConstraintResult.allow()

        if not chunks:
            return ConstraintResult.allow()

        # If query explicitly asks for resolution, allow decisive answer
        if _is_resolution_query(query):
            logger.debug(f"{PIPELINE} ConflictAwareConstraint: resolution query detected, allowing")
            return ConstraintResult.allow()

        # Detect conflicts
        conflicts = _find_conflicts(chunks)

        if not conflicts:
            return ConstraintResult.allow()

        # Format conflict description
        conflict_descriptions = []
        for chunk1_id, class1, chunk2_id, class2 in conflicts[:3]:  # Limit to 3
            conflict_descriptions.append(f"'{class1}' vs '{class2}'")

        reason = f"Conflicting classifications detected: {', '.join(conflict_descriptions)}"

        logger.info(f"{PIPELINE} ConflictAwareConstraint: {reason}")

        return ConstraintResult.deny(
            reason=reason,
            signal="disputed",
            conflicts=conflicts,
            conflict_count=len(conflicts),
        )
