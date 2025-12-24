# fitz_ai/engines/classic_rag/constraints/plugins/causal_attribution.py
"""
Causal Attribution Constraint - Prevents implicit causality claims.

This constraint prevents the system from synthesizing causal explanations
when documents only describe outcomes without explicit causal language.

It enforces: "Don't invent causality that isn't explicitly stated."

This is NOT reasoning suppression. It's epistemic honesty enforcement.
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
# Causal Language Detection
# =============================================================================

# Query patterns that request causal explanation
CAUSAL_QUERY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"\bwhy\b", re.I),
    re.compile(r"\bhow\s+come\b", re.I),
    re.compile(r"\bwhat\s+caused\b", re.I),
    re.compile(r"\bwhat\s+led\s+to\b", re.I),
    re.compile(r"\breason\s+for\b", re.I),
    re.compile(r"\bexplain\b", re.I),
)

# Explicit causal markers in evidence
CAUSAL_EVIDENCE_MARKERS: tuple[str, ...] = (
    "because",
    "due to",
    "caused by",
    "led to",
    "resulted in",
    "resulted from",
    "as a result",
    "consequence of",
    "owing to",
    "on account of",
    "reason is",
    "reason was",
    "reason being",
    "therefore",
    "thus",
    "hence",
    "so that",
    "in order to",
    "stems from",
    "attributed to",
    "triggered by",
    "brought about",
    "gave rise to",
)


def _is_causal_query(query: str) -> bool:
    """Detect if query requests causal explanation."""
    for pattern in CAUSAL_QUERY_PATTERNS:
        if pattern.search(query):
            return True
    return False


def _has_causal_evidence(chunks: Sequence["Chunk"]) -> bool:
    """Check if any chunk contains explicit causal language."""
    for chunk in chunks:
        content_lower = chunk.content.lower()
        for marker in CAUSAL_EVIDENCE_MARKERS:
            if marker in content_lower:
                return True
    return False


def _count_causal_chunks(chunks: Sequence["Chunk"]) -> int:
    """Count chunks containing explicit causal language."""
    count = 0
    for chunk in chunks:
        content_lower = chunk.content.lower()
        for marker in CAUSAL_EVIDENCE_MARKERS:
            if marker in content_lower:
                count += 1
                break  # Count each chunk only once
    return count


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class CausalAttributionConstraint:
    """
    Constraint that prevents implicit causal synthesis.

    When a query requests causal explanation (why, what caused, etc.),
    this constraint verifies that retrieved documents contain explicit
    causal language before allowing a causal answer.

    This prevents the LLM from inventing causal relationships that
    aren't explicitly stated in the evidence.

    Attributes:
        enabled: Whether this constraint is active (default: True)
    """

    enabled: bool = True

    @property
    def name(self) -> str:
        return "causal_attribution"

    def apply(
        self,
        query: str,
        chunks: Sequence["Chunk"],
    ) -> ConstraintResult:
        """
        Check if causal queries have explicit causal evidence.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies causal synthesis if no explicit causal evidence
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Only applies to causal queries
        if not _is_causal_query(query):
            return ConstraintResult.allow()

        # Empty chunks - defer to InsufficientEvidenceConstraint
        if not chunks:
            return ConstraintResult.allow()

        # Check for explicit causal language
        if _has_causal_evidence(chunks):
            logger.debug(
                f"{PIPELINE} CausalAttributionConstraint: causal evidence found"
            )
            return ConstraintResult.allow()

        # Causal query but no explicit causal language in evidence
        causal_count = _count_causal_chunks(chunks)
        logger.info(
            f"{PIPELINE} CausalAttributionConstraint: causal query but no explicit "
            f"causal language in {len(chunks)} chunks"
        )

        return ConstraintResult.deny(
            reason="No explicit causal language found in sources",
            signal="qualified",  # Not abstain - we have evidence, just not causal
            causal_chunks=causal_count,
            total_chunks=len(chunks),
        )
