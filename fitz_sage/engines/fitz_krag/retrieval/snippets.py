"""Bounded source excerpts for retrieval-time scoring."""

from __future__ import annotations

import re

from fitz_sage.engines.fitz_krag.query_planner import content_terms

_MAX_TERM_OCCURRENCES = 32


def query_relevant_excerpt(query: str, text: str, *, max_chars: int) -> str:
    """Return an unchanged source window with the strongest literal query overlap."""
    source = text.strip()
    if not source or max_chars <= 0:
        return ""
    if len(source) <= max_chars:
        return source

    terms = _unique_terms(query)
    if not terms:
        return source[:max_chars].rstrip()

    folded = source.casefold()
    occurrences: dict[str, list[int]] = {}
    frequencies: dict[str, int] = {}
    for term in terms:
        pattern = re.compile(rf"(?<!\w){re.escape(term.casefold())}(?!\w)")
        frequency, positions = _sampled_positions(pattern, folded)
        if not positions:
            continue
        frequencies[term] = frequency
        occurrences[term] = positions

    if not occurrences:
        return source[:max_chars].rstrip()

    anchor_frequency = min(frequencies.values())
    anchor_terms = [
        term for term, frequency in frequencies.items() if frequency == anchor_frequency
    ]
    candidate_starts = {
        _window_start(position, len(source), max_chars)
        for term in anchor_terms
        for position in occurrences[term]
    }

    def score(start: int) -> tuple[int, int, int, int]:
        end = start + max_chars
        matched = [
            term
            for term, positions in occurrences.items()
            if any(start <= position < end for position in positions)
        ]
        total_occurrences = sum(
            sum(start <= position < end for position in positions)
            for positions in occurrences.values()
        )
        return (
            len(matched),
            sum(len(term) for term in matched),
            total_occurrences,
            start,
        )

    best_start = max(candidate_starts, key=score)
    return source[best_start : best_start + max_chars].strip()


def _window_start(position: int, text_length: int, max_chars: int) -> int:
    """Place a match one third into a bounded window when possible."""
    return max(0, min(position - max_chars // 3, text_length - max_chars))


def _sampled_positions(pattern: re.Pattern[str], text: str) -> tuple[int, list[int]]:
    """Return the match count and evenly sampled positions with bounded memory."""
    frequency = sum(1 for _ in pattern.finditer(text))
    if frequency == 0:
        return 0, []
    if frequency <= _MAX_TERM_OCCURRENCES:
        return frequency, [match.start() for match in pattern.finditer(text)]

    last_index = frequency - 1
    sampled_indexes = {
        round(sample * last_index / (_MAX_TERM_OCCURRENCES - 1))
        for sample in range(_MAX_TERM_OCCURRENCES)
    }
    positions = [
        match.start()
        for index, match in enumerate(pattern.finditer(text))
        if index in sampled_indexes
    ]
    return frequency, positions


def _unique_terms(query: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for value in content_terms(query):
        term = value.strip()
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


__all__ = ["query_relevant_excerpt"]
