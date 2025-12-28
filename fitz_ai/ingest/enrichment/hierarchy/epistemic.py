# fitz_ai/ingest/enrichment/hierarchy/epistemic.py
"""
Epistemic assessment for hierarchical summaries.

Reuses the existing constraint plugins for conflict detection,
ensuring a single source of truth for epistemic logic.

Evaluates chunk groups for:
- Internal conflicts/contradictions (via ConflictAwareConstraint)
- Evidence density (sparse/moderate/dense)
- Agreement ratio across sources
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

# Import conflict detection from existing constraint plugin (single source of truth)
from fitz_ai.engines.classic_rag.constraints.plugins.conflict_aware import (
    _find_conflicts as find_constraint_conflicts,
)

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.models.chunk import Chunk


@dataclass
class ConflictInfo:
    """Details about a detected conflict."""

    topic: str
    claim_a: str
    claim_b: str
    chunk_a_id: str
    chunk_b_id: str


@dataclass
class EpistemicAssessment:
    """
    Epistemic status of a chunk group.

    Attributes:
        has_conflicts: Whether contradictory claims were detected
        conflicts: List of specific conflicts found
        conflict_topics: High-level topics where disagreement exists
        agreement_ratio: Fraction of chunks that agree (0.0-1.0)
        evidence_density: How much evidence backs the summary
        chunk_count: Number of source chunks
        total_chars: Total character count of source content
    """

    has_conflicts: bool = False
    conflicts: list[ConflictInfo] = field(default_factory=list)
    conflict_topics: list[str] = field(default_factory=list)
    agreement_ratio: float = 1.0
    evidence_density: str = "moderate"  # "sparse" | "moderate" | "dense"
    chunk_count: int = 0
    total_chars: int = 0

    def to_metadata(self) -> dict:
        """Convert to metadata dict for chunk storage."""
        return {
            "epistemic_has_conflicts": self.has_conflicts,
            "epistemic_conflict_topics": self.conflict_topics,
            "epistemic_agreement_ratio": self.agreement_ratio,
            "epistemic_evidence_density": self.evidence_density,
            "epistemic_chunk_count": self.chunk_count,
        }


# =============================================================================
# Conflict Detection - Uses existing ConflictAwareConstraint logic
# =============================================================================


def _find_conflicts_in_chunks(chunks: Sequence["Chunk"]) -> list[ConflictInfo]:
    """
    Find all conflicts across a set of chunks.

    Delegates to the existing ConflictAwareConstraint's detection logic
    to maintain a single source of truth.
    """
    # Use the existing constraint's conflict detection
    raw_conflicts = find_constraint_conflicts(chunks)

    # Convert to our ConflictInfo format
    conflicts: list[ConflictInfo] = []
    for chunk1_id, class1, chunk2_id, class2 in raw_conflicts:
        conflicts.append(
            ConflictInfo(
                topic="classification",  # The existing constraint focuses on classifications
                claim_a=class1,
                claim_b=class2,
                chunk_a_id=chunk1_id,
                chunk_b_id=chunk2_id,
            )
        )

    return conflicts


# =============================================================================
# Evidence Density
# =============================================================================


def _calculate_evidence_density(chunk_count: int, total_chars: int) -> str:
    """
    Determine evidence density based on chunk count and content size.

    Returns: "sparse", "moderate", or "dense"
    """
    if chunk_count < 3 or total_chars < 500:
        return "sparse"
    elif chunk_count < 10 or total_chars < 3000:
        return "moderate"
    else:
        return "dense"


# =============================================================================
# Public API
# =============================================================================


def assess_chunk_group(chunks: Sequence["Chunk"]) -> EpistemicAssessment:
    """
    Assess the epistemic status of a chunk group before summarization.

    Uses the existing ConflictAwareConstraint for conflict detection,
    ensuring consistency with query-time epistemic checks.

    Args:
        chunks: The chunks to assess

    Returns:
        EpistemicAssessment with conflict and density information
    """
    if not chunks:
        return EpistemicAssessment(
            evidence_density="sparse",
            chunk_count=0,
            total_chars=0,
        )

    # Calculate basic stats
    chunk_count = len(chunks)
    total_chars = sum(len(c.content) for c in chunks)

    # Find conflicts using existing constraint logic
    conflicts = _find_conflicts_in_chunks(chunks)

    # Extract unique conflict topics
    conflict_topics = list({c.topic for c in conflicts})

    # Calculate agreement ratio
    # Simple heuristic: if N chunks and C conflicts, agreement = 1 - (C / N)
    if chunk_count > 1:
        conflict_ratio = min(len(conflicts) / chunk_count, 1.0)
        agreement_ratio = round(1.0 - conflict_ratio, 2)
    else:
        agreement_ratio = 1.0

    # Determine evidence density
    evidence_density = _calculate_evidence_density(chunk_count, total_chars)

    return EpistemicAssessment(
        has_conflicts=len(conflicts) > 0,
        conflicts=conflicts,
        conflict_topics=conflict_topics,
        agreement_ratio=agreement_ratio,
        evidence_density=evidence_density,
        chunk_count=chunk_count,
        total_chars=total_chars,
    )


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


def assess_chunk_group_with_llm(
    chunks: Sequence["Chunk"],
    chat_client: ChatClient,
) -> EpistemicAssessment:
    """
    Assess chunk group using LLM for deeper conflict detection.

    This is slower but catches subtle conflicts that the rule-based
    constraint misses. Use for high-stakes summarization.

    Args:
        chunks: The chunks to assess
        chat_client: LLM client for analysis

    Returns:
        EpistemicAssessment with detailed conflict information
    """
    # Start with constraint-based assessment
    assessment = assess_chunk_group(chunks)

    if len(chunks) < 2:
        return assessment

    # Build content sample for LLM analysis
    content_parts = []
    for i, chunk in enumerate(chunks[:10], 1):  # Limit to 10 chunks
        content = chunk.content[:300]
        if len(chunk.content) > 300:
            content += "..."
        content_parts.append(f"[Source {i}]: {content}")

    content_sample = "\n\n".join(content_parts)

    prompt = f"""Analyze these text passages for contradictions or conflicting claims.

PASSAGES:
{content_sample}

TASK: Identify any contradictions where sources disagree on facts, metrics, trends, or conclusions.

Respond in this exact format:
CONFLICTS_FOUND: yes/no
CONFLICT_TOPICS: comma-separated list (or "none")
AGREEMENT_ESTIMATE: high/medium/low

Only report clear contradictions, not differences in scope or perspective."""

    try:
        response = chat_client.chat([{"role": "user", "content": prompt}])

        # Parse response
        lines = response.strip().split("\n")
        for line in lines:
            if line.startswith("CONFLICTS_FOUND:"):
                has_conflicts = "yes" in line.lower()
                assessment.has_conflicts = has_conflicts or assessment.has_conflicts
            elif line.startswith("CONFLICT_TOPICS:"):
                topics_str = line.split(":", 1)[1].strip()
                if topics_str.lower() != "none":
                    llm_topics = [t.strip() for t in topics_str.split(",")]
                    # Merge with constraint-detected topics
                    all_topics = set(assessment.conflict_topics) | set(llm_topics)
                    assessment.conflict_topics = list(all_topics)
            elif line.startswith("AGREEMENT_ESTIMATE:"):
                estimate = line.split(":", 1)[1].strip().lower()
                if estimate == "low":
                    assessment.agreement_ratio = min(assessment.agreement_ratio, 0.4)
                elif estimate == "medium":
                    assessment.agreement_ratio = min(assessment.agreement_ratio, 0.7)

    except Exception:
        # On LLM failure, fall back to constraint-only assessment
        pass

    return assessment


__all__ = [
    "EpistemicAssessment",
    "ConflictInfo",
    "assess_chunk_group",
    "assess_chunk_group_with_llm",
]
