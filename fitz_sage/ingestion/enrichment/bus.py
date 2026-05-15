# fitz_sage/ingestion/enrichment/bus.py
"""
Chunk Enrichment Bus - Unified per-chunk enrichment with extensible modules.

The enrichment bus runs one LLM call per batch of chunks and extracts
multiple enrichments (summary, keywords, entities, etc.) in a single pass.

Architecture:
    ChunkEnricher
        ├── SummaryModule      → chunk.metadata["summary"]
        ├── KeywordModule      → VocabularyStore
        ├── EntityModule       → chunk.metadata["entities"]
        └── (future modules)   → easily extensible

Usage:
    from fitz_sage.llm import get_chat_factory

    factory = get_chat_factory("cohere")
    enricher = ChunkEnricher(
        chat_factory=factory,
        modules=[SummaryModule(), KeywordModule(), EntityModule()],
    )
    enriched_chunks, keywords = enricher.enrich(chunks)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.ingestion.enrichment.modules import (
    EnrichmentModule,
    EntityModule,
    KeywordModule,
    SummaryModule,
)
from fitz_sage.llm.factory import ChatFactory, ModelTier

if TYPE_CHECKING:
    from fitz_sage.core.chunk import Chunk

logger = logging.getLogger(__name__)


# =============================================================================
# Enrichment Result
# =============================================================================


@dataclass
class ChunkEnrichmentResult:
    """Result of enriching a single chunk."""

    chunk_index: int
    results: dict[str, Any] = field(default_factory=dict)

    def get(self, module_name: str) -> Any:
        """Get result for a specific module."""
        return self.results.get(module_name)


@dataclass
class EnrichmentBatchResult:
    """Result of enriching a batch of chunks."""

    chunks: list["Chunk"]
    all_keywords: list[str] = field(default_factory=list)


# =============================================================================
# Chunk Enricher (The Bus)
# =============================================================================


@dataclass
class ChunkEnricher:
    """
    Unified chunk enrichment bus.

    Runs one LLM call per batch of chunks, extracting all configured
    enrichments in a single pass. Modules are pluggable - add new
    enrichment types by implementing EnrichmentModule.

    Usage:
        from fitz_sage.llm import get_chat_factory

        factory = get_chat_factory("cohere")
        enricher = ChunkEnricher(
            chat_factory=factory,
            modules=[SummaryModule(), KeywordModule(), EntityModule()],
        )
        result = enricher.enrich(chunks)
        # result.chunks have metadata attached
        # result.all_keywords collected for VocabularyStore
    """

    chat_factory: ChatFactory
    modules: list[EnrichmentModule] = field(default_factory=list)
    batch_size: int = 15  # Chunks per LLM call
    max_content_length: int = 1500  # Truncate long chunks
    min_batch_content: int = 500  # Skip LLM call if total batch content below this

    # Tier for enrichment (developer decision - bulk background task)
    TIER_ENRICH: ModelTier = "fast"

    def __post_init__(self) -> None:
        if not self.modules:
            # Default modules if none specified
            self.modules = [SummaryModule(), KeywordModule(), EntityModule()]

    def enrich(self, chunks: list["Chunk"]) -> EnrichmentBatchResult:
        """
        Enrich all chunks with configured modules.

        Args:
            chunks: List of chunks to enrich

        Returns:
            EnrichmentBatchResult with enriched chunks and collected keywords
        """
        if not chunks:
            return EnrichmentBatchResult(chunks=[])

        all_keywords: list[str] = []

        # Process in batches
        for batch_start in range(0, len(chunks), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(chunks))
            batch = chunks[batch_start:batch_end]

            batch_num = batch_start // self.batch_size + 1
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

            logger.info(
                f"[ENRICH] Batch {batch_num}/{total_batches}: enriching {len(batch)} chunks"
            )

            # Run enrichment for this batch
            batch_results = self._enrich_batch(batch)

            # Apply results to chunks and collect keywords
            for chunk, result in zip(batch, batch_results):
                for module in self.modules:
                    module_result = result.get(module.name)
                    if module_result is not None:
                        module.apply_to_chunk(chunk, module_result)

                        # Collect keywords separately
                        if module.name == "keywords" and isinstance(module_result, list):
                            all_keywords.extend(module_result)

        # Deduplicate keywords
        unique_keywords = list(dict.fromkeys(all_keywords))

        logger.info(
            f"[ENRICH] Completed: {len(chunks)} chunks, {len(unique_keywords)} unique keywords"
        )

        return EnrichmentBatchResult(chunks=chunks, all_keywords=unique_keywords)

    def _enrich_batch(self, batch: list["Chunk"]) -> list[ChunkEnrichmentResult]:
        """Enrich a single batch of chunks with one LLM call."""
        # Skip LLM call if batch content is too small to be worth it
        total_content = sum(len(chunk.content) for chunk in batch)
        if total_content < self.min_batch_content:
            logger.debug(
                f"[ENRICH] Skipping batch: {total_content} chars < {self.min_batch_content} min"
            )
            return [ChunkEnrichmentResult(chunk_index=i) for i in range(len(batch))]

        prompt = self._build_prompt(batch)
        messages = [{"role": "user", "content": prompt}]

        try:
            chat = self.chat_factory(self.TIER_ENRICH)
            response = chat.chat(messages)
            return self._parse_response(response, len(batch))
        except Exception as e:
            logger.error(f"[ENRICH] LLM call failed: {e}")
            # Return empty results on failure
            return [ChunkEnrichmentResult(chunk_index=i) for i in range(len(batch))]

    def _build_prompt(self, batch: list["Chunk"]) -> str:
        """Build the combined prompt for a batch."""
        # Build module instructions
        module_instructions = ",\n  ".join(module.prompt_instruction() for module in self.modules)

        prompt_parts = [
            "Analyze each numbered chunk and extract structured information.\n\n"
            "For each chunk, return a JSON object with:\n"
            "{\n"
            f"  {module_instructions}\n"
            "}\n\n"
            "Return a JSON array with one object per chunk, in order.\n"
            "Example response format:\n"
            "```json\n"
            "[\n"
            '  {"summary": "...", "keywords": [...], "entities": [...]},\n'
            '  {"summary": "...", "keywords": [...], "entities": [...]}\n'
            "]\n"
            "```\n\n"
            "CHUNKS TO ANALYZE:\n"
        ]

        for i, chunk in enumerate(batch, 1):
            source = chunk.metadata.get("source_file", chunk.doc_id)
            file_name = Path(source).name if source else f"chunk_{i}"

            # Truncate content
            content = chunk.content
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "\n... [truncated]"

            prompt_parts.append(f"\n--- CHUNK [{i}] from {file_name} ---\n{content}\n")

        prompt_parts.append(
            "\n--- END OF CHUNKS ---\n\nNow return the JSON array with analysis for each chunk:"
        )

        return "".join(prompt_parts)

    def _parse_response(self, response: str, expected_count: int) -> list[ChunkEnrichmentResult]:
        """Parse the LLM response into enrichment results."""
        results: list[ChunkEnrichmentResult] = []

        # Try to extract JSON from response
        json_data = parse_llm_json(response, as_array=True)

        if isinstance(json_data, list):
            for i, item in enumerate(json_data):
                if i >= expected_count:
                    break
                result = ChunkEnrichmentResult(chunk_index=i)
                if isinstance(item, dict):
                    for module in self.modules:
                        if module.json_key in item:
                            parsed = module.parse_result(item[module.json_key])
                            result.results[module.name] = parsed
                results.append(result)

        # Fill in missing results
        while len(results) < expected_count:
            results.append(ChunkEnrichmentResult(chunk_index=len(results)))

        return results


# =============================================================================
# Factory
# =============================================================================


def create_default_enricher(
    chat_factory: ChatFactory,
    min_batch_content: int = 500,
) -> ChunkEnricher:
    """Create an enricher with default modules.

    Args:
        chat_factory: Chat factory for per-task tier selection
        min_batch_content: Minimum total content in batch to trigger LLM call (default: 500 chars)
    """
    return ChunkEnricher(
        chat_factory=chat_factory,
        modules=[
            SummaryModule(),
            KeywordModule(),
            EntityModule(),
        ],
        min_batch_content=min_batch_content,
    )


__all__ = [
    "ChunkEnricher",
    "ChunkEnrichmentResult",
    "EnrichmentBatchResult",
    "create_default_enricher",
]
