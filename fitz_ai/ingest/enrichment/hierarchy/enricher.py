# fitz_ai/ingest/enrichment/hierarchy/enricher.py
"""
Hierarchical enrichment implementation.

Orchestrates the filtering, grouping, and summarization of chunks
according to configured hierarchy rules.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, List, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.enrichment.config import HierarchyConfig, HierarchyRule
from fitz_ai.ingest.enrichment.hierarchy.grouper import ChunkGrouper
from fitz_ai.ingest.enrichment.hierarchy.matcher import ChunkMatcher

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
    - Level 2: Original chunks (detail level)
    - Level 1: Group summaries (one per unique group_by value)
    - Level 0: Corpus summary (one per rule)
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

        Marks original chunks with hierarchy_level=2 and generates
        level-1 group summaries and level-0 corpus summaries.

        Args:
            chunks: Original chunks from ingestion

        Returns:
            Original chunks (marked level-2) + generated summary chunks
        """
        if not self._config.enabled or not self._config.rules:
            return chunks

        # Mark all original chunks as level 2 (leaf level)
        for chunk in chunks:
            if "hierarchy_level" not in chunk.metadata:
                chunk.metadata["hierarchy_level"] = 2

        all_summary_chunks: List[Chunk] = []

        for rule in self._config.rules:
            logger.info(f"[HIERARCHY] Processing rule: {rule.name}")
            summary_chunks = self._process_rule(rule, chunks)
            all_summary_chunks.extend(summary_chunks)

        logger.info(
            f"[HIERARCHY] Generated {len(all_summary_chunks)} summary chunks "
            f"from {len(self._config.rules)} rules"
        )

        return chunks + all_summary_chunks

    def _process_rule(
        self,
        rule: HierarchyRule,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        """Process a single hierarchy rule."""
        # Step 1: Filter chunks by path patterns
        matcher = ChunkMatcher(rule.paths)
        filtered = matcher.filter_chunks(chunks)

        if not filtered:
            logger.info(
                f"[HIERARCHY] No chunks matched patterns for rule '{rule.name}'"
            )
            return []

        logger.info(
            f"[HIERARCHY] Rule '{rule.name}': {len(filtered)}/{len(chunks)} chunks matched"
        )

        # Step 2: Group by metadata key
        grouper = ChunkGrouper(rule.group_by)
        groups = grouper.group(filtered)

        # Step 3: Generate level-1 summaries for each group
        level1_chunks: List[Chunk] = []
        for group_key, group_chunks in groups.items():
            if group_key == "_ungrouped":
                logger.debug(f"[HIERARCHY] Skipping _ungrouped for rule '{rule.name}'")
                continue

            summary_chunk = self._generate_group_summary(
                rule=rule,
                group_key=group_key,
                chunks=group_chunks,
            )
            level1_chunks.append(summary_chunk)

        # Step 4: Generate level-0 corpus summary
        level0_chunk = self._generate_corpus_summary(
            rule=rule,
            level1_chunks=level1_chunks,
        )

        result = level1_chunks
        if level0_chunk:
            result.append(level0_chunk)

        return result

    def _generate_group_summary(
        self,
        rule: HierarchyRule,
        group_key: str,
        chunks: List[Chunk],
    ) -> Chunk:
        """Generate a level-1 summary for a group of chunks."""
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

{rule.prompt}

CONTENT:
{context}

Write a comprehensive summary (2-4 paragraphs) that captures the key information.
"""

        messages = [{"role": "user", "content": prompt}]
        summary_content = self._chat.chat(messages)

        # Create summary chunk
        chunk_id = hashlib.sha256(
            f"hierarchy:{rule.name}:{group_key}".encode()
        ).hexdigest()[:16]

        return Chunk(
            id=f"hierarchy_l1:{chunk_id}",
            doc_id=f"hierarchy:{rule.name}",
            content=summary_content,
            chunk_index=0,
            metadata={
                "hierarchy_level": 1,
                "hierarchy_rule": rule.name,
                "hierarchy_group": group_key,
                rule.group_by: group_key,
                "source_chunk_count": len(chunks),
                "is_hierarchy_summary": True,
            },
        )

    def _generate_corpus_summary(
        self,
        rule: HierarchyRule,
        level1_chunks: List[Chunk],
    ) -> Chunk | None:
        """Generate a level-0 corpus summary from level-1 summaries."""
        if not level1_chunks:
            return None

        # Build context from level-1 summaries
        context_parts = []
        for chunk in level1_chunks:
            group_key = chunk.metadata.get("hierarchy_group", "unknown")
            context_parts.append(f"## {group_key}\n{chunk.content}")

        context = "\n\n".join(context_parts)

        corpus_prompt = rule.corpus_prompt or (
            "Synthesize an overview from the following group summaries. "
            "Identify common themes, patterns, and notable differences."
        )

        prompt = f"""You are creating a corpus-level overview for a knowledge base.

RULE: {rule.name}
GROUPS: {len(level1_chunks)} summaries

{corpus_prompt}

GROUP SUMMARIES:
{context}

Write a high-level overview (3-5 paragraphs) synthesizing the key insights.
"""

        messages = [{"role": "user", "content": prompt}]
        summary_content = self._chat.chat(messages)

        chunk_id = hashlib.sha256(f"hierarchy:corpus:{rule.name}".encode()).hexdigest()[
            :16
        ]

        return Chunk(
            id=f"hierarchy_l0:{chunk_id}",
            doc_id=f"hierarchy:corpus:{rule.name}",
            content=summary_content,
            chunk_index=0,
            metadata={
                "hierarchy_level": 0,
                "hierarchy_rule": rule.name,
                "is_hierarchy_summary": True,
                "is_corpus_summary": True,
                "source_group_count": len(level1_chunks),
            },
        )


__all__ = ["HierarchyEnricher", "ChatClient"]
