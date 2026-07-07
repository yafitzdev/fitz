# fitz_sage/engines/fitz_krag/ingestion/enricher.py
"""
KRAG-specific enrichment — extract keywords and entities from symbols/sections.

Reuses the LLM-based enrichment pattern from the shared enrichment bus but
adapted for KRAG's data model (symbol dicts + section dicts instead of Chunks).
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from fitz_sage.core.json_utils import parse_llm_json

if TYPE_CHECKING:
    from fitz_sage.llm.providers.base import ChatProvider

logger = logging.getLogger(__name__)

_IDENTIFIER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b[A-Z][A-Z0-9]{1,12}-[A-Z0-9-]*\d[A-Z0-9-]*\b"),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+){0,2}\b", re.IGNORECASE),
    re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b"),
    re.compile(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+\b"),
    re.compile(r"(?:/[A-Za-z0-9._{}:-]+){2,}"),
)
_ENTITY_PHRASE_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:[ \t]+[A-Z][A-Za-z0-9]+){1,4}\b")
_ENTITY_STOP_PHRASES = {
    "Abstract",
    "Chapter",
    "Content",
    "Figure",
    "Section",
    "Table",
}
_MIN_ENRICHMENT_TOKENS = 256
_ENRICHMENT_TOKENS_PER_ITEM = 128
_MAX_ENRICHMENT_TOKENS = 2048
_RETRY_ENRICHMENT_TOKENS = 2048


class EnrichmentStrategy(str, Enum):
    """LLM enrichment payloads supported by the KRAG enrichment bus."""

    KEYWORDS = "keywords"
    ENTITIES = "entities"
    FULL = "full"


class KragEnricher:
    """Batch LLM enrichment for KRAG symbols and sections."""

    def __init__(self, chat: "ChatProvider", batch_size: int = 15):
        self._chat = chat
        self._batch_size = batch_size
        self._fallback_only = False

    def enrich_symbols(self, symbol_dicts: list[dict[str, Any]]) -> None:
        """Enrich symbol dicts in-place with keywords, entities, and temporal metadata."""
        self._enrich_symbol_dicts(symbol_dicts, EnrichmentStrategy.FULL)

    def enrich_symbol_keywords(self, symbol_dicts: list[dict[str, Any]]) -> None:
        """Enrich symbol dicts in-place with retrieval keywords only."""
        self._enrich_symbol_dicts(symbol_dicts, EnrichmentStrategy.KEYWORDS)

    def enrich_symbol_entities(self, symbol_dicts: list[dict[str, Any]]) -> None:
        """Enrich symbol dicts in-place with entities and temporal metadata."""
        self._enrich_symbol_dicts(symbol_dicts, EnrichmentStrategy.ENTITIES)

    def enrich_sections(self, section_dicts: list[dict[str, Any]]) -> None:
        """Enrich section dicts in-place with keywords, entities, and temporal metadata."""
        self._enrich_section_dicts(section_dicts, EnrichmentStrategy.FULL)

    def enrich_section_keywords(self, section_dicts: list[dict[str, Any]]) -> None:
        """Enrich section dicts in-place with retrieval keywords only."""
        self._enrich_section_dicts(section_dicts, EnrichmentStrategy.KEYWORDS)

    def enrich_section_entities(self, section_dicts: list[dict[str, Any]]) -> None:
        """Enrich section dicts in-place with entities and temporal metadata."""
        self._enrich_section_dicts(section_dicts, EnrichmentStrategy.ENTITIES)

    def derive_section_entities(self, section_dicts: list[dict[str, Any]]) -> None:
        """Derive document-section entities without a second generation pass."""
        for section in section_dicts:
            item = {
                "name": section.get("title", ""),
                "content": (section.get("summary", "") or section.get("content", ""))[:1000],
            }
            section["entities"] = _merge_entities(
                section.get("entities", []),
                _deterministic_entities(item),
            )

    def _enrich_symbol_dicts(
        self,
        symbol_dicts: list[dict[str, Any]],
        strategy: EnrichmentStrategy,
    ) -> None:
        """Apply one enrichment strategy to symbol dicts."""
        for i in range(0, len(symbol_dicts), self._batch_size):
            batch = symbol_dicts[i : i + self._batch_size]
            items = [
                {
                    "name": s.get("name", ""),
                    "content": s.get("summary", "") or f"{s.get('kind', '')} {s.get('name', '')}",
                }
                for s in batch
            ]
            enriched = self._enrich_batch(items, strategy)
            for j, enrichment in enumerate(enriched):
                _apply_enrichment(batch[j], items[j], enrichment, strategy)

    def _enrich_section_dicts(
        self,
        section_dicts: list[dict[str, Any]],
        strategy: EnrichmentStrategy,
    ) -> None:
        """Apply one enrichment strategy to section dicts."""
        for i in range(0, len(section_dicts), self._batch_size):
            batch = section_dicts[i : i + self._batch_size]
            items = [
                {
                    "name": s.get("title", ""),
                    "content": (s.get("summary", "") or s.get("content", ""))[:500],
                }
                for s in batch
            ]
            enriched = self._enrich_batch(items, strategy)
            for j, enrichment in enumerate(enriched):
                _apply_enrichment(batch[j], items[j], enrichment, strategy)

    def _enrich_batch(
        self,
        items: list[dict[str, str]],
        strategy: EnrichmentStrategy,
    ) -> list[dict[str, Any]]:
        """Run a single LLM call to extract keywords + entities for a batch."""
        if self._fallback_only:
            return _deterministic_enrichments(items, strategy)

        parts = []
        for i, item in enumerate(items):
            parts.append(
                f'<item index="{i + 1}">\n'
                f"name: {item['name']}\n"
                "content:\n"
                f"{item['content']}\n"
                "</item>"
            )
        prompt = "\n\n".join(parts)

        try:
            messages = [
                {
                    "role": "system",
                    "content": _strategy_prompt(strategy, len(items)),
                },
                {"role": "user", "content": prompt},
            ]
            response = self._chat.chat(
                messages,
                max_tokens=_enrichment_max_tokens(len(items)),
                temperature=0,
            )
            try:
                return self._parse_response(response, len(items))
            except ValueError:
                retry_response = self._chat.chat(
                    [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "Retry the same enrichment. The previous response was not valid "
                                f"JSON for exactly {len(items)} item block(s). Return only the "
                                "JSON array, with no markdown and no extra objects."
                            ),
                        },
                    ],
                    max_tokens=_RETRY_ENRICHMENT_TOKENS,
                    temperature=0,
                )
                try:
                    return self._parse_response(retry_response, len(items))
                except ValueError:
                    logger.warning(
                        "Enrichment model returned invalid JSON twice; "
                        "using deterministic fallback for %s item(s).",
                        len(items),
                    )
                    return _deterministic_enrichments(items, strategy)
        except Exception as e:
            self._fallback_only = True
            logger.warning(
                "Enrichment model call failed; using deterministic fallback for %s item(s): %s",
                len(items),
                e,
            )
            return _deterministic_enrichments(items, strategy)

    def _parse_response(self, response: str, expected_count: int) -> list[dict[str, Any]]:
        """Parse LLM response into list of enrichment dicts."""
        parsed = parse_llm_json(response, as_array=True)
        enrichments = _valid_enrichment_list(parsed, expected_count)
        if enrichments is not None:
            return enrichments

        parsed_object = parse_llm_json(response, as_array=False)
        if expected_count == 1 and _looks_like_enrichment(parsed_object):
            return [_normalize_enrichment(parsed_object)]
        if isinstance(parsed_object, dict):
            for key in ("items", "results", "enrichments", "data"):
                enrichments = _valid_enrichment_list(parsed_object.get(key), expected_count)
                if enrichments is not None:
                    return enrichments

        if expected_count == 1:
            salvaged = _salvage_single_enrichment(response)
            if salvaged is not None:
                return [salvaged]

        raise ValueError(
            "enrichment model returned invalid JSON; expected a JSON array "
            f"with {expected_count} item(s)"
        )


def _deterministic_keywords(item: dict[str, str]) -> list[str]:
    """Extract exact identifiers that should never depend on model recall."""
    text = f"{item.get('name', '')}\n{item.get('content', '')}"
    keywords: list[str] = []
    seen: set[str] = set()
    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip(".,;:()[]{}")
            if value and value not in seen:
                seen.add(value)
                keywords.append(value)
    return keywords


def _deterministic_entities(item: dict[str, str]) -> list[dict[str, str]]:
    """Extract identifier and named-phrase entities without model generation."""
    text = f"{item.get('name', '')}\n{item.get('content', '')}"
    entities: list[dict[str, str]] = []
    seen: set[str] = set()

    for pattern in _IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).strip(".,;:()[]{}")
            if _add_entity(entities, seen, value, "identifier") and len(entities) >= 8:
                return entities

    for match in _ENTITY_PHRASE_PATTERN.finditer(text):
        value = match.group(0).strip(".,;:()[]{}")
        if value in _ENTITY_STOP_PHRASES:
            continue
        if _add_entity(entities, seen, value, "entity") and len(entities) >= 8:
            return entities

    return entities


def _deterministic_enrichments(
    items: list[dict[str, str]],
    strategy: EnrichmentStrategy,
) -> list[dict[str, Any]]:
    """Build grounded enrichment when model JSON is unusable."""
    enrichments: list[dict[str, Any]] = []
    for item in items:
        enrichment: dict[str, Any] = {}
        if strategy in (EnrichmentStrategy.KEYWORDS, EnrichmentStrategy.FULL):
            enrichment["keywords"] = _deterministic_keywords(item)
        if strategy in (EnrichmentStrategy.ENTITIES, EnrichmentStrategy.FULL):
            enrichment["entities"] = _deterministic_entities(item)
            enrichment["temporal"] = {"dates": [], "versions": [], "refs": []}
        enrichments.append(enrichment)
    return enrichments


def _add_entity(
    entities: list[dict[str, str]],
    seen: set[str],
    value: str,
    entity_type: str,
) -> bool:
    """Append one entity if it is usable and new."""
    if len(value) < 3:
        return False
    key = value.casefold()
    if key in seen:
        return False
    seen.add(key)
    entities.append({"name": value, "type": entity_type})
    return True


def _apply_enrichment(
    target: dict[str, Any],
    item: dict[str, str],
    enrichment: dict[str, Any],
    strategy: EnrichmentStrategy,
) -> None:
    """Apply a parsed enrichment object to one symbol/section dict."""
    if strategy in (EnrichmentStrategy.KEYWORDS, EnrichmentStrategy.FULL):
        target["keywords"] = _merge_keywords(
            enrichment.get("keywords", []),
            _deterministic_keywords(item),
        )

    if strategy in (EnrichmentStrategy.ENTITIES, EnrichmentStrategy.FULL):
        target["entities"] = enrichment.get("entities", [])
        temporal = enrichment.get("temporal")
        if temporal and isinstance(temporal, dict):
            meta = target.get("metadata") or {}
            meta["temporal"] = temporal
            target["metadata"] = meta


def _merge_keywords(model_keywords: Any, deterministic: list[str]) -> list[str]:
    """Merge model and deterministic keywords while preserving first occurrence."""
    merged: list[str] = []
    seen: set[str] = set()
    if isinstance(model_keywords, list):
        for keyword in model_keywords:
            if not isinstance(keyword, str):
                continue
            value = keyword.strip()
            if value and value not in seen:
                seen.add(value)
                merged.append(value)
    for keyword in deterministic:
        if keyword not in seen:
            seen.add(keyword)
            merged.append(keyword)
    return merged


def _merge_entities(
    model_entities: Any, deterministic: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Merge entity dicts while preserving first occurrence."""
    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    if isinstance(model_entities, list):
        for entity in model_entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name or name.casefold() in seen:
                continue
            seen.add(name.casefold())
            merged.append({"name": name, "type": str(entity.get("type", "entity") or "entity")})
    for entity in deterministic:
        name = entity["name"]
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())
        merged.append(entity)
    return merged[:8]


def _enrichment_max_tokens(item_count: int) -> int:
    """Return a bounded generation cap sized for JSON enrichment batches."""
    return min(
        _MAX_ENRICHMENT_TOKENS,
        max(_MIN_ENRICHMENT_TOKENS, item_count * _ENRICHMENT_TOKENS_PER_ITEM),
    )


def _strategy_prompt(strategy: EnrichmentStrategy, item_count: int) -> str:
    """Build the system prompt for one enrichment strategy."""
    common = (
        f"The user message contains exactly {item_count} item block(s). "
        "Each <item> block is one item, even when its content contains "
        "multiple lines, bullets, sentences, or questions.\n"
    )
    if strategy == EnrichmentStrategy.KEYWORDS:
        return common + (
            "Extract retrieval keywords from each item.\n"
            "Keywords: exact-match identifiers, technical terms, IDs, abbreviations, "
            "and short semantic aliases that a user may search for.\n"
            "Use only values grounded in the item text. Return empty arrays if none found.\n\n"
            "Limits per object: at most 8 keywords. Never repeat a keyword. "
            "If uncertain, omit it.\n\n"
            f"Return ONLY a valid JSON array with exactly {item_count} object(s), "
            "one object per <item> block. Do not return markdown fences or prose:\n"
            '[{"keywords": ["<exact term>"]}, ...]'
        )
    if strategy == EnrichmentStrategy.ENTITIES:
        return common + (
            "Extract named entities and temporal references from each item.\n"
            "Entities: named entities with types "
            '(shape: {"name": "<entity>", "type": "<type>"}).\n'
            "Temporal: dates, version numbers, and time references found in the text "
            '(shape: {"dates": [], "versions": [], "refs": []}). '
            "Use only values found in the item text. Return empty arrays if none found.\n\n"
            "Limits per object: at most 6 entities and at most 5 temporal values per "
            "temporal array. Never repeat an entity name. If uncertain, omit it.\n\n"
            f"Return ONLY a valid JSON array with exactly {item_count} object(s), "
            "one object per <item> block. Do not return markdown fences or prose:\n"
            '[{"entities": [{"name": "<entity>", "type": "<type>"}], '
            '"temporal": {"dates": [], "versions": [], "refs": []}}, ...]'
        )
    return common + (
        "Extract keywords, entities, and temporal references from each item.\n"
        "Keywords: exact-match identifiers (function names, class names, "
        "technical terms, IDs, abbreviations).\n"
        "Entities: named entities with types "
        '(shape: {"name": "<entity>", "type": "<type>"}).\n'
        "Temporal: dates, version numbers, and time references found in the text "
        '(shape: {"dates": [], "versions": [], "refs": []}). '
        "Use only values found in the item text. Return empty arrays if none found.\n\n"
        "Limits per object: at most 8 keywords, at most 6 entities, at most "
        "5 temporal values per temporal array. Never repeat a keyword or "
        "entity name. If uncertain, omit it.\n\n"
        f"Return ONLY a valid JSON array with exactly {item_count} object(s), "
        "one object per <item> block. Do not return markdown fences or prose:\n"
        '[{"keywords": ["<exact term>"], '
        '"entities": [{"name": "<entity>", "type": "<type>"}], '
        '"temporal": {"dates": [], "versions": [], "refs": []}}, ...]'
    )


def _valid_enrichment_list(value: Any, expected_count: int) -> list[dict[str, Any]] | None:
    """Return the first expected enrichment objects when the JSON list is usable."""
    if not isinstance(value, list) or len(value) < expected_count:
        return None
    enrichments = value[:expected_count]
    if all(_looks_like_enrichment(item) for item in enrichments):
        return [_normalize_enrichment(item) for item in enrichments]
    return None


def _looks_like_enrichment(value: Any) -> bool:
    """Return whether a parsed JSON object has the enrichment shape."""
    if not isinstance(value, dict):
        return False
    return any(key in value for key in ("keywords", "entities", "temporal"))


def _normalize_enrichment(value: dict[str, Any]) -> dict[str, Any]:
    """Normalize model enrichment into bounded keyword/entity/temporal fields."""
    return {
        "keywords": _coerce_string_list(value.get("keywords"), limit=8),
        "entities": _coerce_entities(value.get("entities"), limit=6),
        "temporal": _coerce_temporal(value.get("temporal")),
    }


def _salvage_single_enrichment(response: str) -> dict[str, Any] | None:
    """Recover usable single-item enrichment from a truncated JSON prefix."""
    keywords = _extract_json_value_after_key(response, "keywords", "[", "]")
    if not isinstance(keywords, list):
        keywords = _extract_complete_strings_from_array(response, "keywords")
    entities = _extract_json_value_after_key(response, "entities", "[", "]")
    if not isinstance(entities, list):
        entities = _extract_complete_entity_objects(response)
    temporal = _extract_json_value_after_key(response, "temporal", "{", "}")

    normalized = _normalize_enrichment(
        {
            "keywords": keywords,
            "entities": entities,
            "temporal": temporal,
        }
    )
    if normalized["keywords"] or normalized["entities"]:
        return normalized
    return None


def _extract_complete_strings_from_array(text: str, key: str) -> list[str]:
    """Extract complete JSON strings from a truncated string array."""
    key_index = text.find(f'"{key}"')
    if key_index < 0:
        return []
    array_start = text.find("[", key_index)
    if array_start < 0:
        return []

    values: list[str] = []
    index = array_start + 1
    while index < len(text):
        if text[index] == "]":
            break
        if text[index] != '"':
            index += 1
            continue

        end = index + 1
        escaped = False
        while end < len(text):
            char = text[end]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                break
            end += 1
        if end >= len(text) or text[end] != '"':
            break
        try:
            value = json.loads(text[index : end + 1])
        except json.JSONDecodeError:
            break
        if isinstance(value, str):
            values.append(value)
        index = end + 1
    return values


def _extract_json_value_after_key(
    text: str,
    key: str,
    opener: str,
    closer: str,
) -> Any:
    """Extract a complete JSON value following a top-level response key."""
    key_index = text.find(f'"{key}"')
    if key_index < 0:
        return None
    colon_index = text.find(":", key_index)
    start = text.find(opener, colon_index)
    if colon_index < 0 or start < 0:
        return None
    end = _find_balanced_end(text, start, opener, closer)
    if end is None:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _extract_complete_entity_objects(text: str) -> list[dict[str, Any]]:
    """Extract complete entity dicts from an otherwise truncated entities array."""
    key_index = text.find('"entities"')
    if key_index < 0:
        return []
    array_start = text.find("[", key_index)
    if array_start < 0:
        return []

    entities: list[dict[str, Any]] = []
    index = array_start + 1
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        end = _find_balanced_end(text, index, "{", "}")
        if end is None:
            break
        try:
            entity = json.loads(text[index : end + 1])
        except json.JSONDecodeError:
            break
        if isinstance(entity, dict):
            entities.append(entity)
        index = end + 1
    return entities


def _find_balanced_end(text: str, start: int, opener: str, closer: str) -> int | None:
    """Return the matching closer index for a JSON array/object prefix."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return index
    return None


def _coerce_string_list(value: Any, *, limit: int) -> list[str]:
    """Coerce a model value into a bounded unique string list."""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
        if len(result) >= limit:
            break
    return result


def _coerce_entities(value: Any, *, limit: int) -> list[dict[str, str]]:
    """Coerce model entities into bounded unique name/type dicts."""
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "unknown")).strip() or "unknown"
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            result.append({"name": name, "type": entity_type})
        if len(result) >= limit:
            break
    return result


def _coerce_temporal(value: Any) -> dict[str, list[str]]:
    """Coerce temporal metadata into stable bounded arrays."""
    temporal = value if isinstance(value, dict) else {}
    return {
        "dates": _coerce_string_list(temporal.get("dates"), limit=5),
        "versions": _coerce_string_list(temporal.get("versions"), limit=5),
        "refs": _coerce_string_list(temporal.get("refs"), limit=5),
    }
