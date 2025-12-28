# fitz_ai/ingest/enrichment/hierarchy/enricher.py
"""
Hierarchical enrichment implementation.

Supports two modes:
1. Simple mode (zero-config): Just enable hierarchy, uses smart defaults
2. Rules mode (power-user): Configure custom rules for complex scenarios

Simple mode groups chunks by source file and generates summaries
using prompts from the centralized prompt library.

Epistemic features:
- Detects conflicts within chunk groups before summarization
- Adapts prompts to acknowledge uncertainty and disagreement
- Propagates epistemic metadata up the hierarchy
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.enrichment.config import (
    HierarchyConfig,
    HierarchyRule,
)
from fitz_ai.ingest.enrichment.hierarchy.epistemic import (
    EpistemicAssessment,
    assess_chunk_group,
)
from fitz_ai.ingest.enrichment.hierarchy.grouper import ChunkGrouper
from fitz_ai.ingest.enrichment.hierarchy.matcher import ChunkMatcher
from fitz_ai.prompts import hierarchy as hierarchy_prompts

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@runtime_checkable
class ChatClient(Protocol):
    """Protocol for LLM chat clients."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class HierarchyEnricher:
    """
    Generates hierarchical summaries from chunks.

    For each configured rule:
    1. Filters chunks by path patterns
    2. Groups by metadata key
    3. Generates level-1 summaries for each group
    4. Generates level-0 corpus summary

    All summary chunks are returned alongside original chunks with
    `hierarchy_level` metadata marking their position in the hierarchy.

    Hierarchy levels:
    - Level 0: Original chunks (detail level)
    - Level 1: Group summaries (one per unique group_by value)
    - Level 2: Corpus summary (one per rule)
    """

    def __init__(
        self,
        config: HierarchyConfig,
        chat_client: ChatClient,
    ):
        """
        Initialize the hierarchy enricher.

        Args:
            config: Hierarchy configuration with rules
            chat_client: LLM client for generating summaries
        """
        self._config = config
        self._chat = chat_client

    def enrich(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Apply hierarchical enrichment to chunks.

        Supports two modes:
        1. Simple mode: No rules configured, uses config.group_by and default prompts
        2. Rules mode: Custom rules for power users

        Marks original chunks with hierarchy_level=0 and generates
        level-1 group summaries and level-2 corpus summaries.

        Args:
            chunks: Original chunks from ingestion

        Returns:
            Original chunks (marked level-0) + generated summary chunks
        """
        if not self._config.enabled:
            return chunks

        # Mark all original chunks as level 0 (leaf level)
        for chunk in chunks:
            if "hierarchy_level" not in chunk.metadata:
                chunk.metadata["hierarchy_level"] = 0

        all_summary_chunks: List[Chunk] = []

        if self._config.rules:
            # Power-user mode: process configured rules
            for rule in self._config.rules:
                logger.info(f"[HIERARCHY] Processing rule: {rule.name}")
                summary_chunks = self._process_rule(rule, chunks)
                all_summary_chunks.extend(summary_chunks)

            logger.info(
                f"[HIERARCHY] Generated {len(all_summary_chunks)} summary chunks "
                f"from {len(self._config.rules)} rules"
            )
        else:
            # Simple mode: use defaults, no path filtering
            logger.info(f"[HIERARCHY] Simple mode: grouping by '{self._config.group_by}'")
            summary_chunks = self._process_simple_mode(chunks)
            all_summary_chunks.extend(summary_chunks)

            logger.info(
                f"[HIERARCHY] Generated {len(all_summary_chunks)} summary chunks in simple mode"
            )

        return chunks + all_summary_chunks

    def _process_simple_mode(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Process chunks in simple mode (no rules, use defaults).

        Groups all chunks by the configured group_by key and generates
        summaries using default prompts. Includes epistemic assessment
        to detect conflicts and uncertainty.
        """
        if not chunks:
            return []

        # Group by configured key (default: "source")
        grouper = ChunkGrouper(self._config.group_by)
        groups = grouper.group(chunks)

        # Get prompts (use prompt library defaults if not specified)
        group_prompt = self._config.group_prompt or hierarchy_prompts.GROUP_SUMMARY_PROMPT

        # Generate level-1 summaries for each group, tracking epistemic assessments
        level1_chunks: List[Chunk] = []
        group_assessments: List[EpistemicAssessment] = []

        for group_key, group_chunks in groups.items():
            if group_key == "_ungrouped":
                logger.debug("[HIERARCHY] Skipping _ungrouped in simple mode")
                continue

            summary_chunk, assessment = self._generate_simple_group_summary(
                group_key=group_key,
                chunks=group_chunks,
                prompt=group_prompt,
            )
            level1_chunks.append(summary_chunk)
            group_assessments.append(assessment)

        # Generate level-2 corpus summary with epistemic context
        level2_chunk = self._generate_simple_corpus_summary(
            level1_chunks=level1_chunks,
            group_assessments=group_assessments,
        )

        result = level1_chunks
        if level2_chunk:
            result.append(level2_chunk)

        return result

    def _generate_simple_group_summary(
        self,
        group_key: str,
        chunks: List[Chunk],
        prompt: str,
    ) -> tuple[Chunk, EpistemicAssessment]:
        """Generate a level-1 summary for a group in simple mode."""
        # Assess epistemic status before summarizing
        assessment = assess_chunk_group(chunks)

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

        chunk_id = hashlib.sha256(f"hierarchy:simple:{group_key}".encode()).hexdigest()[:16]

        # Include epistemic metadata
        metadata = {
            "hierarchy_level": 1,
            "hierarchy_rule": "_simple",
            "hierarchy_group": group_key,
            self._config.group_by: group_key,
            "source_chunk_count": len(chunks),
            "is_hierarchy_summary": True,
        }
        metadata.update(assessment.to_metadata())

        return (
            Chunk(
                id=f"hierarchy_l1:{chunk_id}",
                doc_id="hierarchy:simple",
                content=summary_content,
                chunk_index=0,
                metadata=metadata,
            ),
            assessment,
        )

    def _generate_simple_corpus_summary(
        self,
        level1_chunks: List[Chunk],
        group_assessments: List[EpistemicAssessment],
    ) -> Chunk | None:
        """Generate a level-2 corpus summary in simple mode with epistemic awareness."""
        if not level1_chunks:
            return None

        # Build epistemic-aware prompt based on group assessments
        effective_prompt = hierarchy_prompts.build_epistemic_corpus_prompt(group_assessments)

        # Build epistemic context block
        epistemic_context = hierarchy_prompts.build_epistemic_corpus_context(group_assessments)

        # Build context from level-1 summaries
        context_parts = []
        for chunk in level1_chunks:
            group_key = chunk.metadata.get("hierarchy_group", "unknown")
            context_parts.append(f"## {group_key}\n{chunk.content}")

        context = "\n\n".join(context_parts)

        llm_prompt = f"""You are creating a corpus-level overview for a knowledge base.

GROUPS: {len(level1_chunks)} summaries

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
                "source_group_count": len(level1_chunks),
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
    ) -> List[Chunk]:
        """Process a single hierarchy rule with epistemic awareness."""
        # Step 1: Filter chunks by path patterns
        matcher = ChunkMatcher(rule.paths)
        filtered = matcher.filter_chunks(chunks)

        if not filtered:
            logger.info(f"[HIERARCHY] No chunks matched patterns for rule '{rule.name}'")
            return []

        logger.info(f"[HIERARCHY] Rule '{rule.name}': {len(filtered)}/{len(chunks)} chunks matched")

        # Step 2: Group by metadata key
        grouper = ChunkGrouper(rule.group_by)
        groups = grouper.group(filtered)

        # Step 3: Generate level-1 summaries for each group, tracking assessments
        level1_chunks: List[Chunk] = []
        group_assessments: List[EpistemicAssessment] = []

        for group_key, group_chunks in groups.items():
            if group_key == "_ungrouped":
                logger.debug(f"[HIERARCHY] Skipping _ungrouped for rule '{rule.name}'")
                continue

            summary_chunk, assessment = self._generate_group_summary(
                rule=rule,
                group_key=group_key,
                chunks=group_chunks,
            )
            level1_chunks.append(summary_chunk)
            group_assessments.append(assessment)

        # Step 4: Generate level-2 corpus summary with epistemic context
        level2_chunk = self._generate_corpus_summary(
            rule=rule,
            level1_chunks=level1_chunks,
            group_assessments=group_assessments,
        )

        result = level1_chunks
        if level2_chunk:
            result.append(level2_chunk)

        return result

    def _generate_group_summary(
        self,
        rule: HierarchyRule,
        group_key: str,
        chunks: List[Chunk],
    ) -> tuple[Chunk, EpistemicAssessment]:
        """Generate a level-1 summary for a group of chunks with epistemic awareness."""
        # Assess epistemic status before summarizing
        assessment = assess_chunk_group(chunks)

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

        # Create summary chunk with epistemic metadata
        chunk_id = hashlib.sha256(f"hierarchy:{rule.name}:{group_key}".encode()).hexdigest()[:16]

        metadata = {
            "hierarchy_level": 1,
            "hierarchy_rule": rule.name,
            "hierarchy_group": group_key,
            rule.group_by: group_key,
            "source_chunk_count": len(chunks),
            "is_hierarchy_summary": True,
        }
        metadata.update(assessment.to_metadata())

        return (
            Chunk(
                id=f"hierarchy_l1:{chunk_id}",
                doc_id=f"hierarchy:{rule.name}",
                content=summary_content,
                chunk_index=0,
                metadata=metadata,
            ),
            assessment,
        )

    def _generate_corpus_summary(
        self,
        rule: HierarchyRule,
        level1_chunks: List[Chunk],
        group_assessments: List[EpistemicAssessment],
    ) -> Chunk | None:
        """Generate a level-2 corpus summary from level-1 summaries with epistemic awareness."""
        if not level1_chunks:
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
        for chunk in level1_chunks:
            group_key = chunk.metadata.get("hierarchy_group", "unknown")
            context_parts.append(f"## {group_key}\n{chunk.content}")

        context = "\n\n".join(context_parts)

        prompt = f"""You are creating a corpus-level overview for a knowledge base.

RULE: {rule.name}
GROUPS: {len(level1_chunks)} summaries

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
                "source_group_count": len(level1_chunks),
                "epistemic_groups_with_conflicts": groups_with_conflicts,
                "epistemic_conflict_topics": list(all_conflict_topics),
                "epistemic_agreement_ratio": round(avg_agreement, 2),
                "epistemic_evidence_density": corpus_density,
            },
        )


__all__ = ["HierarchyEnricher", "ChatClient"]
