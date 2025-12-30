# fitz_ai/ingestion/enrichment/hierarchy/enricher.py
"""
Hierarchical enrichment implementation.

Supports two modes:
1. Simple mode (zero-config): Just enable hierarchy, uses smart defaults
2. Rules mode (power-user): Configure custom rules for complex scenarios

Simple mode groups chunks by source file and generates summaries
using prompts from the centralized prompt library.

Storage model:
- L0 (original chunks): Enriched with hierarchy_summary metadata containing their group's L1 summary
- L1 (group summaries): NOT stored as separate chunks - only as metadata on L0 chunks
- L2 (corpus summary): The ONLY separate chunk created - summarizes all L1 summaries

Epistemic features:
- Detects conflicts using core/conflicts.py (platform-wide capability)
- Adapts prompts to acknowledge uncertainty and disagreement
- Propagates epistemic metadata up the hierarchy
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

from fitz_ai.core.chunk import Chunk
from fitz_ai.core.conflicts import find_conflicts
from fitz_ai.core.guardrails.semantic import SemanticMatcher
from fitz_ai.ingestion.enrichment.config import (
    HierarchyConfig,
    HierarchyRule,
)
from fitz_ai.ingestion.enrichment.hierarchy.grouper import ChunkGrouper
from fitz_ai.ingestion.enrichment.hierarchy.matcher import ChunkMatcher
from fitz_ai.prompts import hierarchy as hierarchy_prompts

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


# =============================================================================
# Epistemic Assessment (inlined from epistemic.py)
# =============================================================================


@dataclass
class EpistemicAssessment:
    """
    Epistemic status of a chunk group.

    Attributes:
        has_conflicts: Whether contradictory claims were detected
        conflict_topics: High-level topics where disagreement exists
        agreement_ratio: Fraction of chunks that agree (0.0-1.0)
        evidence_density: How much evidence backs the summary
        chunk_count: Number of source chunks
    """

    has_conflicts: bool = False
    conflict_topics: list[str] = field(default_factory=list)
    agreement_ratio: float = 1.0
    evidence_density: str = "moderate"  # "sparse" | "moderate" | "dense"
    chunk_count: int = 0

    def to_metadata(self) -> dict:
        """Convert to metadata dict for chunk storage."""
        return {
            "epistemic_has_conflicts": self.has_conflicts,
            "epistemic_conflict_topics": self.conflict_topics,
            "epistemic_agreement_ratio": self.agreement_ratio,
            "epistemic_evidence_density": self.evidence_density,
            "epistemic_chunk_count": self.chunk_count,
        }


def assess_chunk_group(
    chunks: List[Chunk],
    semantic_matcher: SemanticMatcher | None = None,
) -> EpistemicAssessment:
    """
    Assess the epistemic status of a chunk group before summarization.

    Args:
        chunks: List of chunks to assess
        semantic_matcher: Optional SemanticMatcher for semantic conflict detection.
                         If not provided, falls back to legacy find_conflicts() stub.

    Returns:
        EpistemicAssessment with conflict and density information
    """
    if not chunks:
        return EpistemicAssessment(evidence_density="sparse", chunk_count=0)

    chunk_count = len(chunks)
    total_chars = sum(len(c.content) for c in chunks)

    # Find conflicts using semantic matcher if available, otherwise stub
    if semantic_matcher is not None:
        conflicts = semantic_matcher.find_conflicts(chunks)
    else:
        conflicts = find_conflicts(chunks)

    # Calculate agreement ratio
    if chunk_count > 1 and conflicts:
        conflict_ratio = min(len(conflicts) / chunk_count, 1.0)
        agreement_ratio = round(1.0 - conflict_ratio, 2)
    else:
        agreement_ratio = 1.0

    # Determine evidence density
    if chunk_count < 3 or total_chars < 500:
        evidence_density = "sparse"
    elif chunk_count < 10 or total_chars < 3000:
        evidence_density = "moderate"
    else:
        evidence_density = "dense"

    return EpistemicAssessment(
        has_conflicts=len(conflicts) > 0,
        conflict_topics=["claims"] if conflicts else [],
        agreement_ratio=agreement_ratio,
        evidence_density=evidence_density,
        chunk_count=chunk_count,
    )


# =============================================================================
# Hierarchy Enricher
# =============================================================================


class HierarchyEnricher:
    """
    Generates hierarchical summaries from chunks.

    For each configured rule:
    1. Filters chunks by path patterns
    2. Groups by metadata key
    3. Generates level-1 summaries for each group (stored as metadata on original chunks)
    4. Generates level-2 corpus summary (the only separate chunk created)

    Storage model:
    - Original chunks are enriched with `hierarchy_summary` metadata containing their group's L1 summary
    - Only the L2 corpus summary is stored as a separate retrievable chunk
    - This prevents summaries from crowding out source documents in retrieval

    Hierarchy levels:
    - Level 0: Original chunks with hierarchy_summary metadata
    - Level 1: Group summaries (metadata only, not separate chunks)
    - Level 2: Corpus summary (one separate chunk per rule)
    """

    def __init__(
        self,
        config: HierarchyConfig,
        chat_client: ChatClient,
        semantic_matcher: SemanticMatcher | None = None,
    ):
        """
        Initialize the hierarchy enricher.

        Args:
            config: Hierarchy configuration with rules
            chat_client: LLM client for generating summaries
            semantic_matcher: Optional SemanticMatcher for conflict detection.
                             If not provided, conflicts won't be detected.
        """
        self._config = config
        self._chat = chat_client
        self._semantic_matcher = semantic_matcher

    def enrich(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply hierarchical enrichment to chunks.

        Supports two modes:
        1. Simple mode: No rules configured, uses config.group_by and default prompts
        2. Rules mode: Custom rules for power users

        L1 summaries are stored as metadata on original chunks.
        Only L2 corpus summary is created as a separate chunk.

        Args:
            chunks: Original chunks from ingestion

        Returns:
            Original chunks (enriched with hierarchy_summary metadata) + L2 corpus summary chunk
        """
        if not self._config.enabled:
            return chunks

        # Mark all original chunks as level 0 (leaf level)
        for chunk in chunks:
            if "hierarchy_level" not in chunk.metadata:
                chunk.metadata["hierarchy_level"] = 0

        corpus_chunks: List[Chunk] = []

        if self._config.rules:
            # Power-user mode: process configured rules
            for rule in self._config.rules:
                logger.info(f"[HIERARCHY] Processing rule: {rule.name}")
                corpus_chunk = self._process_rule(rule, chunks)
                if corpus_chunk:
                    corpus_chunks.append(corpus_chunk)

            logger.info(
                f"[HIERARCHY] Generated {len(corpus_chunks)} corpus summary chunks "
                f"from {len(self._config.rules)} rules"
            )
        else:
            # Simple mode: use defaults, no path filtering
            logger.info(f"[HIERARCHY] Simple mode: grouping by '{self._config.group_by}'")
            corpus_chunk = self._process_simple_mode(chunks)
            if corpus_chunk:
                corpus_chunks.append(corpus_chunk)

            logger.info(
                f"[HIERARCHY] Generated {len(corpus_chunks)} corpus summary chunk in simple mode"
            )

        return chunks + corpus_chunks

    def _process_simple_mode(self, chunks: List[Chunk]) -> Chunk | None:
        """
        Process chunks in simple mode (no rules, use defaults).

        Groups all chunks by the configured group_by key and generates
        summaries using default prompts. L1 summaries are stored as metadata
        on the original chunks. Only returns the L2 corpus summary chunk.
        """
        if not chunks:
            return None

        # Group by configured key (default: "source")
        grouper = ChunkGrouper(self._config.group_by)
        groups = grouper.group(chunks)

        # Get prompts (use prompt library defaults if not specified)
        group_prompt = self._config.group_prompt or hierarchy_prompts.GROUP_SUMMARY_PROMPT

        # Generate level-1 summaries for each group, storing as metadata on chunks
        level1_summaries: List[str] = []
        group_assessments: List[EpistemicAssessment] = []
        group_keys: List[str] = []

        for group_key, group_chunks in groups.items():
            if group_key == "_ungrouped":
                logger.debug("[HIERARCHY] Skipping _ungrouped in simple mode")
                continue

            summary_content, assessment = self._generate_simple_group_summary(
                group_key=group_key,
                chunks=group_chunks,
                prompt=group_prompt,
            )

            # Store L1 summary as metadata on each chunk in this group
            for chunk in group_chunks:
                chunk.metadata["hierarchy_summary"] = summary_content
                chunk.metadata["hierarchy_group"] = group_key
                chunk.metadata.update(assessment.to_metadata())

            level1_summaries.append(summary_content)
            group_assessments.append(assessment)
            group_keys.append(group_key)

        # Generate level-2 corpus summary with epistemic context
        level2_chunk = self._generate_simple_corpus_summary(
            level1_summaries=level1_summaries,
            group_keys=group_keys,
            group_assessments=group_assessments,
        )

        return level2_chunk

    def _generate_simple_group_summary(
        self,
        group_key: str,
        chunks: List[Chunk],
        prompt: str,
    ) -> tuple[str, EpistemicAssessment]:
        """Generate a level-1 summary for a group in simple mode.

        Returns:
            Tuple of (summary_content, assessment) - summary is stored as metadata, not as a chunk.
        """
        # Assess epistemic status before summarizing
        assessment = assess_chunk_group(chunks, semantic_matcher=self._semantic_matcher)

        if assessment.has_conflicts:
            logger.info(
                f"[HIERARCHY] Group '{group_key}' has conflicts on: {assessment.conflict_topics}"
            )

        # Build epistemic-aware prompt
        if assessment.has_conflicts or assessment.evidence_density == "sparse":
            effective_prompt = hierarchy_prompts.build_epistemic_group_prompt(assessment)
        else:
            effective_prompt = prompt

        # Build context from chunks (limit to avoid token overflow)
        context_parts = []
        max_chunks = 20
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            content = chunk.content[:500]
            if len(chunk.content) > 500:
                content += "..."
            context_parts.append(f"[{i}] {content}")

        if len(chunks) > max_chunks:
            context_parts.append(f"... and {len(chunks) - max_chunks} more items")

        context = "\n\n".join(context_parts)

        llm_prompt = f"""You are summarizing content for a knowledge base.

GROUP: {group_key}
ITEMS: {len(chunks)}

{effective_prompt}

CONTENT:
{context}

Write a comprehensive summary (2-4 paragraphs) that captures the key information.
"""

        messages = [{"role": "user", "content": llm_prompt}]
        summary_content = self._chat.chat(messages)

        return (summary_content, assessment)

    def _generate_simple_corpus_summary(
        self,
        level1_summaries: List[str],
        group_keys: List[str],
        group_assessments: List[EpistemicAssessment],
    ) -> Chunk | None:
        """Generate a level-2 corpus summary in simple mode with epistemic awareness.

        This is the ONLY separate chunk created by the hierarchy enricher.
        L1 summaries are passed as strings (they're stored as metadata on original chunks).
        """
        if not level1_summaries:
            return None

        # Build epistemic-aware prompt based on group assessments
        effective_prompt = hierarchy_prompts.build_epistemic_corpus_prompt(group_assessments)

        # Build epistemic context block
        epistemic_context = hierarchy_prompts.build_epistemic_corpus_context(group_assessments)

        # Build context from level-1 summaries
        context_parts = []
        for group_key, summary in zip(group_keys, level1_summaries):
            context_parts.append(f"## {group_key}\n{summary}")

        context = "\n\n".join(context_parts)

        llm_prompt = f"""You are creating a corpus-level overview for a knowledge base.

GROUPS: {len(level1_summaries)} summaries

{effective_prompt}

{epistemic_context}
GROUP SUMMARIES:
{context}

Write a high-level overview (3-5 paragraphs) synthesizing the key insights.
"""

        messages = [{"role": "user", "content": llm_prompt}]
        summary_content = self._chat.chat(messages)

        chunk_id = hashlib.sha256("hierarchy:corpus:simple".encode()).hexdigest()[:16]

        # Aggregate epistemic metadata for corpus level
        groups_with_conflicts = sum(1 for a in group_assessments if a.has_conflicts)
        all_conflict_topics: set[str] = set()
        for assessment in group_assessments:
            all_conflict_topics.update(assessment.conflict_topics)

        avg_agreement = (
            sum(a.agreement_ratio for a in group_assessments) / len(group_assessments)
            if group_assessments
            else 1.0
        )

        sparse_count = sum(1 for a in group_assessments if a.evidence_density == "sparse")
        corpus_density = (
            "sparse"
            if sparse_count > len(group_assessments) / 2
            else "moderate"
            if sparse_count > 0
            else "dense"
        )

        return Chunk(
            id=f"hierarchy_l2:{chunk_id}",
            doc_id="hierarchy:corpus:simple",
            content=summary_content,
            chunk_index=0,
            metadata={
                "hierarchy_level": 2,
                "hierarchy_rule": "_simple",
                "is_hierarchy_summary": True,
                "is_corpus_summary": True,
                "source_group_count": len(level1_summaries),
                "epistemic_groups_with_conflicts": groups_with_conflicts,
                "epistemic_conflict_topics": list(all_conflict_topics),
                "epistemic_agreement_ratio": round(avg_agreement, 2),
                "epistemic_evidence_density": corpus_density,
            },
        )

    def _process_rule(
        self,
        rule: HierarchyRule,
        chunks: List[Chunk],
    ) -> Chunk | None:
        """Process a single hierarchy rule with epistemic awareness.

        L1 summaries are stored as metadata on original chunks.
        Only returns the L2 corpus summary chunk.
        """
        # Step 1: Filter chunks by path patterns
        matcher = ChunkMatcher(rule.paths)
        filtered = matcher.filter_chunks(chunks)

        if not filtered:
            logger.info(f"[HIERARCHY] No chunks matched patterns for rule '{rule.name}'")
            return None

        logger.info(f"[HIERARCHY] Rule '{rule.name}': {len(filtered)}/{len(chunks)} chunks matched")

        # Step 2: Group by metadata key
        grouper = ChunkGrouper(rule.group_by)
        groups = grouper.group(filtered)

        # Step 3: Generate level-1 summaries for each group, storing as metadata
        level1_summaries: List[str] = []
        group_assessments: List[EpistemicAssessment] = []
        group_keys: List[str] = []

        for group_key, group_chunks in groups.items():
            if group_key == "_ungrouped":
                logger.debug(f"[HIERARCHY] Skipping _ungrouped for rule '{rule.name}'")
                continue

            summary_content, assessment = self._generate_group_summary(
                rule=rule,
                group_key=group_key,
                chunks=group_chunks,
            )

            # Store L1 summary as metadata on each chunk in this group
            for chunk in group_chunks:
                chunk.metadata["hierarchy_summary"] = summary_content
                chunk.metadata["hierarchy_group"] = group_key
                chunk.metadata["hierarchy_rule"] = rule.name
                chunk.metadata.update(assessment.to_metadata())

            level1_summaries.append(summary_content)
            group_assessments.append(assessment)
            group_keys.append(group_key)

        # Step 4: Generate level-2 corpus summary with epistemic context
        level2_chunk = self._generate_corpus_summary(
            rule=rule,
            level1_summaries=level1_summaries,
            group_keys=group_keys,
            group_assessments=group_assessments,
        )

        return level2_chunk

    def _generate_group_summary(
        self,
        rule: HierarchyRule,
        group_key: str,
        chunks: List[Chunk],
    ) -> tuple[str, EpistemicAssessment]:
        """Generate a level-1 summary for a group of chunks with epistemic awareness.

        Returns:
            Tuple of (summary_content, assessment) - summary is stored as metadata, not as a chunk.
        """
        # Assess epistemic status before summarizing
        assessment = assess_chunk_group(chunks, semantic_matcher=self._semantic_matcher)

        if assessment.has_conflicts:
            logger.info(
                f"[HIERARCHY] Rule '{rule.name}' group '{group_key}' "
                f"has conflicts on: {assessment.conflict_topics}"
            )

        # Build epistemic-aware prompt if needed
        if assessment.has_conflicts or assessment.evidence_density == "sparse":
            effective_prompt = hierarchy_prompts.build_epistemic_group_prompt(assessment)
            # Append rule-specific prompt if provided
            if rule.prompt:
                effective_prompt = f"{effective_prompt}\n\nADDITIONAL GUIDANCE:\n{rule.prompt}"
        else:
            effective_prompt = rule.prompt

        # Build context from chunks (limit to avoid token overflow)
        context_parts = []
        max_chunks = 20
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            # Truncate individual chunks to reasonable length
            content = chunk.content[:500]
            if len(chunk.content) > 500:
                content += "..."
            context_parts.append(f"[{i}] {content}")

        if len(chunks) > max_chunks:
            context_parts.append(f"... and {len(chunks) - max_chunks} more items")

        context = "\n\n".join(context_parts)

        prompt = f"""You are summarizing content for a knowledge base.

GROUP: {group_key}
ITEMS: {len(chunks)}

{effective_prompt}

CONTENT:
{context}

Write a comprehensive summary (2-4 paragraphs) that captures the key information.
"""

        messages = [{"role": "user", "content": prompt}]
        summary_content = self._chat.chat(messages)

        return (summary_content, assessment)

    def _generate_corpus_summary(
        self,
        rule: HierarchyRule,
        level1_summaries: List[str],
        group_keys: List[str],
        group_assessments: List[EpistemicAssessment],
    ) -> Chunk | None:
        """Generate a level-2 corpus summary from level-1 summaries with epistemic awareness.

        This is the ONLY separate chunk created by the hierarchy enricher for this rule.
        L1 summaries are passed as strings (they're stored as metadata on original chunks).
        """
        if not level1_summaries:
            return None

        # Build epistemic-aware prompt based on group assessments
        effective_prompt = hierarchy_prompts.build_epistemic_corpus_prompt(group_assessments)

        # Append rule-specific corpus prompt if provided
        if rule.corpus_prompt:
            effective_prompt = f"{effective_prompt}\n\nADDITIONAL GUIDANCE:\n{rule.corpus_prompt}"

        # Build epistemic context block
        epistemic_context = hierarchy_prompts.build_epistemic_corpus_context(group_assessments)

        # Build context from level-1 summaries
        context_parts = []
        for group_key, summary in zip(group_keys, level1_summaries):
            context_parts.append(f"## {group_key}\n{summary}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are creating a corpus-level overview for a knowledge base.

RULE: {rule.name}
GROUPS: {len(level1_summaries)} summaries

{effective_prompt}

{epistemic_context}
GROUP SUMMARIES:
{context}

Write a high-level overview (3-5 paragraphs) synthesizing the key insights.
"""

        messages = [{"role": "user", "content": prompt}]
        summary_content = self._chat.chat(messages)

        chunk_id = hashlib.sha256(f"hierarchy:corpus:{rule.name}".encode()).hexdigest()[:16]

        # Aggregate epistemic metadata for corpus level
        groups_with_conflicts = sum(1 for a in group_assessments if a.has_conflicts)
        all_conflict_topics: set[str] = set()
        for assessment in group_assessments:
            all_conflict_topics.update(assessment.conflict_topics)

        avg_agreement = (
            sum(a.agreement_ratio for a in group_assessments) / len(group_assessments)
            if group_assessments
            else 1.0
        )

        sparse_count = sum(1 for a in group_assessments if a.evidence_density == "sparse")
        corpus_density = (
            "sparse"
            if sparse_count > len(group_assessments) / 2
            else "moderate"
            if sparse_count > 0
            else "dense"
        )

        return Chunk(
            id=f"hierarchy_l2:{chunk_id}",
            doc_id=f"hierarchy:corpus:{rule.name}",
            content=summary_content,
            chunk_index=0,
            metadata={
                "hierarchy_level": 2,
                "hierarchy_rule": rule.name,
                "is_hierarchy_summary": True,
                "is_corpus_summary": True,
                "source_group_count": len(level1_summaries),
                "epistemic_groups_with_conflicts": groups_with_conflicts,
                "epistemic_conflict_topics": list(all_conflict_topics),
                "epistemic_agreement_ratio": round(avg_agreement, 2),
                "epistemic_evidence_density": corpus_density,
            },
        )


__all__ = ["HierarchyEnricher", "ChatClient", "EpistemicAssessment", "assess_chunk_group"]
