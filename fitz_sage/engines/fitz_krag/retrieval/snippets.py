"""Bounded source excerpts for retrieval-time scoring."""

from __future__ import annotations

import re

from fitz_sage.engines.fitz_krag.query_planner import content_terms

_MAX_QUERY_TERMS = 32
_MAX_ANCHOR_TERMS = 8
_POSITION_BUCKETS = 32
_MAX_CANDIDATE_WINDOWS = 128


def query_relevant_excerpt(query: str, text: str, *, max_chars: int) -> str:
    """Return an unchanged source window with strong literal query overlap."""
    source = text.strip()
    if not source or max_chars <= 0:
        return ""
    if len(source) <= max_chars:
        return source

    terms = _query_terms(query)
    if not terms:
        return source[:max_chars].rstrip()

    frequencies, occurrences = _term_occurrences(terms, source)
    if not occurrences:
        return source[:max_chars].rstrip()

    anchor_frequency = min(frequencies.values())
    anchor_terms = sorted(
        (
            term
            for term, frequency in frequencies.items()
            if frequency == anchor_frequency
        ),
        key=lambda term: (-len(term), terms.index(term)),
    )[:_MAX_ANCHOR_TERMS]
    candidate_starts = _bounded_candidate_starts(
        {
            _window_start(position, len(source), max_chars)
            for term in anchor_terms
            for position in occurrences[term]
        }
    )

    def score(start: int) -> tuple[int, int, int, int, int, int]:
        end = start + max_chars
        local_positions = {
            term: [position for position in positions if start <= position < end]
            for term, positions in occurrences.items()
        }
        local_positions = {
            term: positions for term, positions in local_positions.items() if positions
        }
        flattened = [position for positions in local_positions.values() for position in positions]
        span = max(flattened) - min(flattened) if len(flattened) > 1 else 0
        return (
            len(local_positions),
            sum(1_000_000 // frequencies[term] for term in local_positions),
            sum(len(term) for term in local_positions),
            len(flattened),
            -span,
            -start,
        )

    best_start = max(candidate_starts, key=score)
    return source[best_start : best_start + max_chars].strip()


def _window_start(position: int, text_length: int, max_chars: int) -> int:
    """Place a match one third into a bounded window when possible."""
    return max(0, min(position - max_chars // 3, text_length - max_chars))


def _term_occurrences(
    terms: list[str],
    text: str,
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Collect bounded, document-wide positions for all terms in one scan."""
    alternatives = "|".join(
        re.escape(term)
        for term in sorted(terms, key=lambda value: (-len(value), terms.index(value)))
    )
    pattern = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)
    frequencies = {term: 0 for term in terms}
    buckets: dict[str, dict[int, tuple[int, int]]] = {term: {} for term in terms}
    text_length = max(1, len(text))

    for match in pattern.finditer(text):
        term = match.group(0).casefold()
        if term not in frequencies:
            continue
        position = match.start()
        frequencies[term] += 1
        bucket = min(_POSITION_BUCKETS - 1, position * _POSITION_BUCKETS // text_length)
        first, _ = buckets[term].get(bucket, (position, position))
        buckets[term][bucket] = (first, position)

    observed_frequencies = {
        term: frequency for term, frequency in frequencies.items() if frequency > 0
    }
    occurrences = {
        term: sorted(
            {
                position
                for first_last in buckets[term].values()
                for position in first_last
            }
        )
        for term in observed_frequencies
    }
    return observed_frequencies, occurrences


def _bounded_candidate_starts(starts: set[int]) -> list[int]:
    """Evenly sample candidate windows while retaining both document ends."""
    ordered = sorted(starts)
    if len(ordered) <= _MAX_CANDIDATE_WINDOWS:
        return ordered
    last_index = len(ordered) - 1
    indexes = {
        round(sample * last_index / (_MAX_CANDIDATE_WINDOWS - 1))
        for sample in range(_MAX_CANDIDATE_WINDOWS)
    }
    return [ordered[index] for index in sorted(indexes)]


def _query_terms(query: str) -> list[str]:
    """Return a bounded set of high-information literal query terms."""
    unique: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for index, value in enumerate(content_terms(query)):
        term = value.strip()
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        unique.append((index, term, key))

    ranked = sorted(
        unique,
        key=lambda item: (
            not (
                "_" in item[1]
                or (
                    any(character.isdigit() for character in item[1])
                    and any(character.isupper() for character in item[1])
                )
            ),
            -len(item[1]),
            item[0],
        ),
    )
    selected = ranked[:_MAX_QUERY_TERMS]
    return [key for _, _, key in selected]


__all__ = ["query_relevant_excerpt"]
