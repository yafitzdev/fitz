"""Query-leg provenance and bounded coverage preservation."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any

from fitz_sage.engines.fitz_krag.types import Address

_RETRIEVAL_QUERIES_KEY = "retrieval_queries"


def tag_retrieval_query(addresses: list[Address], query: str) -> list[Address]:
    """Record which concrete BM25 query produced each address."""
    query = query.strip()
    if not query:
        return addresses

    tagged: list[Address] = []
    for address in addresses:
        metadata = dict(address.metadata)
        queries = _query_values(metadata)
        if query.casefold() in {value.casefold() for value in queries}:
            tagged.append(address)
            continue
        metadata[_RETRIEVAL_QUERIES_KEY] = [*queries, query]
        tagged.append(replace(address, metadata=metadata))
    return tagged


def merge_retrieval_provenance(primary: Address, duplicate: Address) -> Address:
    """Merge query and temporal provenance for duplicate addresses."""
    metadata = dict(primary.metadata)
    changed = False
    for key in (_RETRIEVAL_QUERIES_KEY, "temporal_refs"):
        existing = _string_values(metadata.get(key))
        merged = _merge_case_insensitive(existing, _string_values(duplicate.metadata.get(key)))
        if merged != existing:
            metadata[key] = merged
            changed = True

    score = max(primary.score, duplicate.score)
    if not changed and score == primary.score:
        return primary
    return replace(primary, score=score, metadata=metadata)


def ensure_query_coverage(
    candidates: list[Address],
    selected: list[Address],
    required_queries: list[str],
    *,
    limit: int,
) -> list[Address]:
    """Keep one candidate from each successful query leg within a fixed limit."""
    if limit <= 0 or not candidates or not required_queries:
        return selected[: max(0, limit)]

    required = _ordered_query_keys(required_queries)
    if not required:
        return selected[:limit]

    output = list(selected[:limit])
    selected_ids = {_address_identity(address) for address in output}
    coverage = Counter(
        query_key for address in output for query_key in _covered_query_keys(address, required)
    )

    for query_key in required:
        if coverage[query_key] > 0:
            continue
        candidate = next(
            (
                address
                for address in candidates
                if query_key in _covered_query_keys(address, required)
                and _address_identity(address) not in selected_ids
            ),
            None,
        )
        if candidate is None:
            continue

        if len(output) < limit:
            output.append(candidate)
        else:
            replace_at = _replaceable_index(output, coverage, required)
            if replace_at is None:
                continue
            removed = output[replace_at]
            selected_ids.remove(_address_identity(removed))
            for covered in _covered_query_keys(removed, required):
                coverage[covered] -= 1
            output[replace_at] = candidate

        selected_ids.add(_address_identity(candidate))
        for covered in _covered_query_keys(candidate, required):
            coverage[covered] += 1

    return output


def compound_queries(rewrite_result: Any) -> list[str]:
    """Return explicit decomposition legs from a rewrite result."""
    if not rewrite_result or not bool(getattr(rewrite_result, "is_compound", False)):
        return []
    values = getattr(rewrite_result, "decomposed_queries", [])
    return _merge_case_insensitive([], _string_values(values))


def _replaceable_index(
    selected: list[Address],
    coverage: Counter[str],
    required: list[str],
) -> int | None:
    for index in range(len(selected) - 1, -1, -1):
        covered = _covered_query_keys(selected[index], required)
        if not covered or all(coverage[key] > 1 for key in covered):
            return index
    return None


def _covered_query_keys(address: Address, required: list[str]) -> set[str]:
    provenance = {value.casefold() for value in _query_values(address.metadata)}
    return {query_key for query_key in required if query_key in provenance}


def _ordered_query_keys(queries: list[str]) -> list[str]:
    return [value.casefold() for value in _merge_case_insensitive([], queries)]


def _query_values(metadata: dict[str, Any]) -> list[str]:
    return _string_values(metadata.get(_RETRIEVAL_QUERIES_KEY))


def _string_values(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value if (text := str(item).strip())]


def _merge_case_insensitive(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = {value.casefold() for value in existing}
    for value in additions:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(value)
    return merged


def _address_identity(address: Address) -> tuple[str, str]:
    return address.source_id, address.location


__all__ = [
    "compound_queries",
    "ensure_query_coverage",
    "merge_retrieval_provenance",
    "tag_retrieval_query",
]
