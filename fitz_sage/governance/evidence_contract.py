# fitz_sage/governance/evidence_contract.py
"""Pyrrho-owned evidence contract projection for retrieval governance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "inside",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_QUESTION_TITLE_WORDS = {
    "Can",
    "Compare",
    "Does",
    "For",
    "How",
    "Is",
    "Using",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
}

EXACT_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])_?[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+"
    r"(?:[-_][A-Za-z0-9]+)*(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])(?=[A-Za-z0-9-]*\d)[A-Za-z][A-Za-z0-9]*"
    r"(?:-[A-Za-z0-9]+)+(?![A-Za-z0-9_])|"
    r"\b[A-Z]{2,}[A-Z0-9]*\d[A-Z0-9_-]*\b|"
    r"\b[A-Z]\d+\b"
)

_MODALITY_TO_ADDRESS_KINDS = {
    "code": ("symbol",),
    "configuration": ("symbol", "section"),
    "structured_table": ("table",),
    "unstructured_text": ("section",),
    "log_trace": ("section",),
    "pdf_layout": ("section",),
    "mixed": (),
}


@dataclass(frozen=True)
class QueryContract:
    """Evidence obligations projected from Pyrrho g4-alpha query heads."""

    query: str
    query_contract: str | None = None
    route: str | None = None
    answerability_shape: str | None = None
    retrieval_modality: str | None = None
    identifiers: tuple[str, ...] = ()
    phrase_anchors: tuple[str, ...] = ()
    source_anchors: tuple[str, ...] = ()
    keyword_anchors: tuple[str, ...] = ()
    metric_terms: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = ()
    temporal_policy: str | None = None

    @property
    def has_hard_anchors(self) -> bool:
        """Return whether literal anchors must be represented in evidence."""
        return bool(self.identifiers or self.phrase_anchors)


def build_query_contract(query: str, profile: Any = None) -> QueryContract:
    """Build the evidence contract from Pyrrho heads plus literal anchors.

    Pyrrho owns semantic obligations. This function does not infer table/code,
    temporal, source-authority, metric, conflict, or coverage policy from query
    wording. Query text contributes only literal anchors used for mechanical
    matching against retrieved evidence.
    """
    query_contract = _profile_value(profile, "query_contract")
    answerability_shape = _profile_value(profile, "answerability_shape")
    retrieval_modality = _profile_value(profile, "retrieval_modality")
    return QueryContract(
        query=query,
        query_contract=query_contract,
        route=_profile_value(profile, "query_route"),
        answerability_shape=answerability_shape,
        retrieval_modality=retrieval_modality,
        identifiers=tuple(exact_identifiers(query)),
        phrase_anchors=tuple(_capitalized_phrases(query)),
        keyword_anchors=tuple(_keyword_anchors(query)),
        required_modalities=_required_modalities_from_pyrrho(retrieval_modality),
        temporal_policy="temporal" if query_contract == "temporal_grounding" else None,
    )


def exact_identifiers(query: str) -> list[str]:
    """Extract literal exact identifiers from query text."""
    identifiers: list[str] = []
    seen: set[str] = set()
    for match in EXACT_IDENTIFIER_PATTERN.finditer(query):
        value = match.group(0).strip(".,;:()[]{}")
        if value.lower() in seen:
            continue
        seen.add(value.lower())
        identifiers.append(value)
    return identifiers


def normalize_text(value: str) -> str:
    """Normalize text for mechanical evidence matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _required_modalities_from_pyrrho(retrieval_modality: str | None) -> tuple[str, ...]:
    """Map Pyrrho's retrieval modality head to address kinds."""
    if retrieval_modality is None:
        return ()
    return _MODALITY_TO_ADDRESS_KINDS.get(retrieval_modality, ())


def _profile_value(profile: Any, name: str) -> str | None:
    """Read a trusted Pyrrho-derived profile value."""
    value = getattr(profile, name, None)
    return str(value) if value else None


def _capitalized_phrases(query: str) -> list[str]:
    """Extract literal title-cased anchors such as Project Nebula."""
    phrases: list[str] = []
    for match in re.finditer(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+\b", query):
        words = match.group(0).strip().split()
        while words and words[0] in _QUESTION_TITLE_WORDS:
            words = words[1:]
        phrase = " ".join(words)
        if len(words) < 2:
            continue
        if phrase.lower() in {"return material authorization"}:
            continue
        if phrase and phrase.lower() not in {item.lower() for item in phrases}:
            phrases.append(phrase)
    return phrases


def _keyword_anchors(query: str) -> list[str]:
    """Extract lexical query terms used only for mechanical evidence alignment."""
    normalized = normalize_text(query)
    tokens = [
        token
        for token in normalized.split()
        if (len(token) >= 3 or re.fullmatch(r"[a-z]\d+", token)) and token not in _STOPWORDS
    ]
    return list(dict.fromkeys(tokens))


__all__ = [
    "EXACT_IDENTIFIER_PATTERN",
    "QueryContract",
    "build_query_contract",
    "exact_identifiers",
    "normalize_text",
]
