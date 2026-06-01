# fitz_sage/engines/fitz_krag/ingestion/enricher.py
"""
KRAG-specific enrichment — extract keywords and entities from symbols/sections.

Reuses the LLM-based enrichment pattern from the shared enrichment bus but
adapted for KRAG's data model (symbol dicts + section dicts instead of Chunks).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.core import KnowledgeError
from fitz_sage.core.json_utils import parse_llm_json

if TYPE_CHECKING:
    from fitz_sage.llm.providers.base import ChatProvider

logger = logging.getLogger(__name__)


class KragEnricher:
    """Batch LLM enrichment for KRAG symbols and sections."""

    def __init__(self, chat: "ChatProvider", batch_size: int = 15):
        self._chat = chat
        self._batch_size = batch_size

    def enrich_symbols(self, symbol_dicts: list[dict[str, Any]]) -> None:
        """Enrich symbol dicts in-place with keywords, entities, and temporal metadata."""
        for i in range(0, len(symbol_dicts), self._batch_size):
            batch = symbol_dicts[i : i + self._batch_size]
            items = [
                {
                    "name": s.get("name", ""),
                    "content": s.get("summary", "") or f"{s.get('kind', '')} {s.get('name', '')}",
                }
                for s in batch
            ]
            enriched = self._enrich_batch(items)
            for j, enrichment in enumerate(enriched):
                batch[j]["keywords"] = enrichment.get("keywords", [])
                batch[j]["entities"] = enrichment.get("entities", [])
                temporal = enrichment.get("temporal")
                if temporal and isinstance(temporal, dict):
                    meta = batch[j].get("metadata") or {}
                    meta["temporal"] = temporal
                    batch[j]["metadata"] = meta

    def enrich_sections(self, section_dicts: list[dict[str, Any]]) -> None:
        """Enrich section dicts in-place with keywords, entities, and temporal metadata."""
        for i in range(0, len(section_dicts), self._batch_size):
            batch = section_dicts[i : i + self._batch_size]
            items = [
                {
                    "name": s.get("title", ""),
                    "content": (s.get("summary", "") or s.get("content", ""))[:500],
                }
                for s in batch
            ]
            enriched = self._enrich_batch(items)
            for j, enrichment in enumerate(enriched):
                batch[j]["keywords"] = enrichment.get("keywords", [])
                batch[j]["entities"] = enrichment.get("entities", [])
                temporal = enrichment.get("temporal")
                if temporal and isinstance(temporal, dict):
                    meta = batch[j].get("metadata") or {}
                    meta["temporal"] = temporal
                    batch[j]["metadata"] = meta

    def _enrich_batch(self, items: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Run a single LLM call to extract keywords + entities for a batch."""
        parts = []
        for i, item in enumerate(items):
            parts.append(f"Item {i + 1}: '{item['name']}'\n{item['content']}")
        prompt = "\n\n".join(parts)

        try:
            response = self._chat.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Extract keywords, entities, and temporal references from each item.\n"
                            "Keywords: exact-match identifiers (function names, class names, "
                            "technical terms, IDs, abbreviations).\n"
                            "Entities: named entities with types "
                            '(e.g., {"name": "PostgreSQL", "type": "technology"}).\n'
                            "Temporal: dates, version numbers, and time references found in the text "
                            '(e.g., {"dates": ["2024-03"], "versions": ["v2.3"], "refs": ["latest"]}).'
                            " Return empty object if none found.\n\n"
                            "Return a JSON array with one object per item:\n"
                            '[{"keywords": ["kw1"], "entities": [{"name": "X", "type": "T"}], '
                            '"temporal": {"dates": [], "versions": [], "refs": []}}, ...]'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return self._parse_response(response, len(items))
        except Exception as e:
            logger.error(f"Required enrichment batch failed: {e}")
            raise KnowledgeError(f"Required enrichment batch failed: {e}") from e

    def _parse_response(self, response: str, expected_count: int) -> list[dict[str, Any]]:
        """Parse LLM response into list of enrichment dicts."""
        parsed = parse_llm_json(response, as_array=True)
        if isinstance(parsed, list) and len(parsed) >= expected_count:
            return parsed[:expected_count]
        raise ValueError(
            "enrichment model returned invalid JSON; expected a JSON array "
            f"with {expected_count} item(s)"
        )
