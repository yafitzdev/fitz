# fitz_ai/engines/classic_rag/constraints/plugins/insufficient_evidence.py
"""
Insufficient Evidence Constraint - Default guardrail for evidence coverage.

This constraint prevents the system from giving confident answers when
there is not enough direct evidence in the retrieved chunks.

It does NOT:
- Rank authority
- Resolve ambiguity
- Guess intent
- Use LLM calls

It only enforces: "Is there explicit evidence to justify a decisive answer?"
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
# Query Type Detection
# =============================================================================

# Patterns for different question types
FACT_QUERY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*which\b", re.I),
    re.compile(r"^\s*what\b", re.I),
    re.compile(r"^\s*who\b", re.I),
    re.compile(r"^\s*where\b", re.I),
    re.compile(r"^\s*when\b", re.I),
    re.compile(r"\bwhich\s+\w+\s+(is|was|are|were|does|did|has|have|supports?|caused?)\b", re.I),
)

CAUSAL_QUERY_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^\s*why\b", re.I),
    re.compile(r"^\s*how\s+(did|does|was|were|come)\b", re.I),
    re.compile(r"\bcause\b", re.I),
    re.compile(r"\breason\b", re.I),
    re.compile(r"\bexplain\b", re.I),
)


def _is_fact_query(query: str) -> bool:
    """Detect if query asks for a factual assertion."""
    for pattern in FACT_QUERY_PATTERNS:
        if pattern.search(query):
            return True
    return False


def _is_causal_query(query: str) -> bool:
    """Detect if query asks for causal explanation."""
    for pattern in CAUSAL_QUERY_PATTERNS:
        if pattern.search(query):
            return True
    return False


# =============================================================================
# Evidence Detection
# =============================================================================

# Causal language markers
CAUSAL_MARKERS: tuple[str, ...] = (
    "because",
    "due to",
    "as a result",
    "led to",
    "caused by",
    "owing to",
    "result of",
    "consequence of",
    "reason for",
    "reason is",
    "reason was",
    "therefore",
    "thus",
    "hence",
    "since",  # causal "since", not temporal
    "in order to",
    "so that",
)

# Assertion markers (indicates a definitive statement)
ASSERTION_MARKERS: tuple[str, ...] = (
    " is ",
    " was ",
    " are ",
    " were ",
    " does ",
    " did ",
    " has ",
    " have ",
    " supports ",
    " supported ",
    " caused ",
    " causes ",
    " includes ",
    " included ",
    " provides ",
    " provided ",
    " enables ",
    " enabled ",
)


def _extract_query_keywords(query: str) -> set[str]:
    """Extract key terms from query for matching."""
    # Remove common question words and stopwords
    stopwords = {
        "which", "what", "who", "where", "when", "why", "how",
        "is", "was", "are", "were", "does", "did", "has", "have",
        "the", "a", "an", "of", "in", "on", "for", "to", "and", "or",
        "this", "that", "these", "those", "it", "its",
    }

    words = re.findall(r"\b\w+\b", query.lower())
    keywords = {w for w in words if w not in stopwords and len(w) > 2}

    return keywords


def _chunk_has_causal_evidence(chunk_content: str) -> bool:
    """Check if chunk contains causal language."""
    content_lower = chunk_content.lower()
    for marker in CAUSAL_MARKERS:
        if marker in content_lower:
            return True
    return False


def _chunk_has_assertion(chunk_content: str, query_keywords: set[str]) -> bool:
    """
    Check if chunk contains a direct assertion relevant to query.

    Requires:
    1. At least one query keyword present
    2. An assertion marker (is/was/does/etc.)
    """
    content_lower = chunk_content.lower()

    # Must contain at least one query keyword
    keyword_found = False
    for keyword in query_keywords:
        if keyword in content_lower:
            keyword_found = True
            break

    if not keyword_found:
        return False

    # Must contain an assertion marker
    for marker in ASSERTION_MARKERS:
        if marker in content_lower:
            return True

    return False


def _count_relevant_chunks(
    chunks: Sequence["Chunk"],
    query_keywords: set[str],
    require_causal: bool,
) -> int:
    """Count chunks with relevant evidence."""
    count = 0

    for chunk in chunks:
        content = chunk.content

        if require_causal:
            if _chunk_has_causal_evidence(content):
                count += 1
        else:
            if _chunk_has_assertion(content, query_keywords):
                count += 1

    return count


# =============================================================================
# Constraint Implementation
# =============================================================================


@dataclass
class InsufficientEvidenceConstraint:
    """
    Constraint that prevents confident answers without sufficient evidence.

    This constraint checks:
    1. Are there any chunks at all?
    2. For causal questions: is there causal language?
    3. For fact questions: are there direct assertions?

    Attributes:
        enabled: Whether this constraint is active (default: True)
        min_evidence_count: Minimum chunks with evidence required (default: 1)
    """

    enabled: bool = True
    min_evidence_count: int = 1

    @property
    def name(self) -> str:
        return "insufficient_evidence"

    def apply(
        self,
        query: str,
        chunks: Sequence["Chunk"],
    ) -> ConstraintResult:
        """
        Check if there is sufficient evidence to answer the query.

        Args:
            query: The user's question
            chunks: Retrieved chunks

        Returns:
            ConstraintResult - denies if insufficient evidence
        """
        if not self.enabled:
            return ConstraintResult.allow()

        # Rule 1: Empty context
        if not chunks:
            logger.info(f"{PIPELINE} InsufficientEvidenceConstraint: no chunks retrieved")
            return ConstraintResult.deny(
                reason="No evidence retrieved",
                signal="abstain",
                evidence_count=0,
            )

        # Determine query type
        is_causal = _is_causal_query(query)
        is_fact = _is_fact_query(query)

        # Extract query keywords for matching
        query_keywords = _extract_query_keywords(query)

        # Rule 2 & 3: Check for sufficient evidence
        if is_causal:
            evidence_count = _count_relevant_chunks(
                chunks, query_keywords, require_causal=True
            )

            if evidence_count < self.min_evidence_count:
                logger.info(
                    f"{PIPELINE} InsufficientEvidenceConstraint: "
                    f"causal query but no causal evidence (found {evidence_count})"
                )
                return ConstraintResult.deny(
                    reason="No explicit causal evidence found",
                    signal="abstain",
                    evidence_count=evidence_count,
                    query_type="causal",
                )

        elif is_fact:
            evidence_count = _count_relevant_chunks(
                chunks, query_keywords, require_causal=False
            )

            if evidence_count < self.min_evidence_count:
                logger.info(
                    f"{PIPELINE} InsufficientEvidenceConstraint: "
                    f"fact query but no direct assertion (found {evidence_count})"
                )
                return ConstraintResult.deny(
                    reason="No direct assertion found in retrieved evidence",
                    signal="abstain",
                    evidence_count=evidence_count,
                    query_type="fact",
                )

        # For other query types (or if evidence found), allow
        logger.debug(f"{PIPELINE} InsufficientEvidenceConstraint: sufficient evidence")
        return ConstraintResult.allow()