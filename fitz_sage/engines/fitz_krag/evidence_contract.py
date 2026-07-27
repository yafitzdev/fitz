# fitz_sage/engines/fitz_krag/evidence_contract.py
"""KRAG evidence contract helpers for retrieval and evidence assembly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from fitz_sage.core.identifiers import (
    EXACT_IDENTIFIER_PATTERN,
    contains_exact_identifier,
    exact_identifiers,
)

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
    "Are",
    "Can",
    "Compare",
    "Could",
    "Did",
    "Do",
    "Does",
    "For",
    "How",
    "Is",
    "Should",
    "Using",
    "Was",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Will",
    "Would",
}

_MODALITY_TO_ADDRESS_KINDS = {
    "code": ("symbol",),
    "configuration": ("symbol", "section"),
    "structured_table": ("table",),
    "unstructured_text": ("section",),
    "log_trace": ("section",),
    "pdf_layout": ("section",),
    "mixed": ("section", "table", "symbol"),
}

_OBLIGATION_TO_ADDRESS_KINDS = {
    "row_key_lookup": ("table",),
    "column_value_lookup": ("table",),
    "multi_row_comparison": ("table",),
    "aggregate_or_count": ("table",),
    "stale_row_version": ("table",),
    "symbol_definition": ("symbol",),
    "constant_or_env_var": ("symbol",),
    "call_path_or_helper": ("symbol",),
    "test_or_execution_result": ("symbol",),
    "versioned_api_behavior": ("symbol",),
    "config_key_value": ("symbol", "section"),
    "default_or_fallback": ("symbol", "section"),
    "environment_override": ("symbol", "section"),
    "version_scope": ("symbol", "section"),
    "conflicting_config_sources": ("symbol", "section"),
    "status_or_outcome": ("section",),
    "timestamp_ordering": ("section",),
    "error_signature": ("section",),
    "correlation_id": ("section",),
    "missing_run_result": ("section",),
    "table_or_figure_reference": ("section", "table"),
    "footnote_or_caption": ("section",),
    "section_header_scope": ("section",),
    "page_or_revision_scope": ("section",),
    "form_or_field_value": ("section",),
    "prose_plus_table": ("section", "table"),
    "prose_plus_code": ("section", "symbol"),
    "table_plus_config": ("table", "symbol", "section"),
    "policy_plus_latest_row": ("section", "table"),
    "log_plus_config": ("section", "symbol"),
    "code_plus_changelog": ("symbol", "section"),
}

_V2_EVIDENCE_KIND_TO_ADDRESS_KINDS = {
    "needs_text": ("section",),
    "needs_table_or_record": ("table",),
    "needs_code_or_symbol": ("symbol",),
    "needs_config_or_setting": ("symbol", "section"),
    "needs_log_or_run_result": ("section",),
    "needs_document_layout": ("section",),
}


@dataclass(frozen=True)
class QueryContract:
    """Evidence obligations used by KRAG retrieval and evidence assembly."""

    query: str
    query_contract: str | None = None
    answerability_shape: str | None = None
    retrieval_modality: str | None = None
    retrieval_obligation: str | None = None
    identifiers: tuple[str, ...] = ()
    phrase_anchors: tuple[str, ...] = ()
    keyword_anchors: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = ()
    temporal_policy: str | None = None

    @property
    def has_hard_anchors(self) -> bool:
        """Return whether literal anchors must be represented in evidence."""
        return bool(self.identifiers or self.phrase_anchors)


def build_query_contract(query: str, profile: Any = None) -> QueryContract:
    """Build the executor evidence contract from profile values and anchors.

    Profile values come from KRAG query prep. Query text contributes only
    literal anchors used for mechanical matching against retrieved evidence.
    """
    query_contract = _profile_value(profile, "query_contract")
    answerability_shape = _profile_value(profile, "answerability_shape")
    retrieval_modality = _profile_value(profile, "retrieval_modality")
    retrieval_obligation = _profile_value(profile, "retrieval_obligation")
    required_modalities = _profile_sequence_value(profile, "required_modalities")
    if not required_modalities:
        required_modalities = required_modalities_from_profile(
            retrieval_modality,
            retrieval_obligation,
        )
    return QueryContract(
        query=query,
        query_contract=query_contract,
        answerability_shape=answerability_shape,
        retrieval_modality=retrieval_modality,
        retrieval_obligation=retrieval_obligation,
        identifiers=tuple(exact_identifiers(query)),
        phrase_anchors=tuple(_capitalized_phrases(query)),
        keyword_anchors=tuple(_keyword_anchors(query)),
        required_modalities=required_modalities,
        temporal_policy="temporal" if query_contract == "temporal_grounding" else None,
    )


def normalize_text(value: str) -> str:
    """Normalize text for mechanical evidence matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def required_modalities_from_profile(
    retrieval_modality: str | None,
    retrieval_obligation: str | None = None,
) -> tuple[str, ...]:
    """Translate retrieval profile labels into executor address kinds."""
    obligation_kinds = _OBLIGATION_TO_ADDRESS_KINDS.get(retrieval_obligation or "", ())
    if obligation_kinds:
        return obligation_kinds

    kinds: list[str] = []
    for kind in _MODALITY_TO_ADDRESS_KINDS.get(retrieval_modality or "", ()):
        if kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)


def required_modalities_from_v2_evidence_kinds(
    evidence_kinds: Iterable[str] | None,
) -> tuple[str, ...]:
    """Translate v2 evidence-kind labels into executor address kinds."""
    kinds: list[str] = []
    for evidence_kind in evidence_kinds or ():
        for kind in _V2_EVIDENCE_KIND_TO_ADDRESS_KINDS.get(str(evidence_kind), ()):
            if kind not in kinds:
                kinds.append(kind)
    return tuple(kinds)


def required_modalities_for_obligation(retrieval_obligation: str | None) -> tuple[str, ...]:
    """Return address kinds required by one Pyrrho obligation label."""
    return _OBLIGATION_TO_ADDRESS_KINDS.get(retrieval_obligation or "", ())


def _profile_value(profile: Any, name: str) -> str | None:
    """Read a trusted Pyrrho-derived profile value."""
    value = getattr(profile, name, None)
    return str(value) if value else None


def _profile_sequence_value(profile: Any, name: str) -> tuple[str, ...]:
    """Read a trusted Pyrrho-derived profile sequence value."""
    value = getattr(profile, name, None)
    if value is None or isinstance(value, str):
        return ()
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if item)
    return ()


def _capitalized_phrases(query: str) -> list[str]:
    """Extract literal title-cased anchors such as Project Nebula."""
    phrases: list[str] = []
    for match in re.finditer(r"\b[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+\b", query):
        words = match.group(0).strip().split()
        while words and (words[0] in _QUESTION_TITLE_WORDS or words[0].lower() in _STOPWORDS):
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
    "contains_exact_identifier",
    "exact_identifiers",
    "normalize_text",
    "required_modalities_for_obligation",
    "required_modalities_from_profile",
    "required_modalities_from_v2_evidence_kinds",
]
