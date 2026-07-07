# fitz_sage/engines/fitz_krag/retrieval_profile.py
"""Pyrrho-owned retrieval profile for the KRAG executor."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_sage.governance.evidence_contract import (
    required_modalities_from_pyrrho,
    required_modalities_from_v2_evidence_kinds,
)

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis

_ROUTE_DOMAINS = {
    "technology_computing": "technical",
    "law_policy": "legal",
    "science_medicine": "medical",
    "economics_finance": "financial",
    "history_geography": "general",
    "culture_society": "general",
    "general_commonsense": "general",
}

_HEAD_NAMES = (
    "query_contract",
    "route",
    "answerability_shape",
    "retrieval_modality",
    "retrieval_obligation",
    "retrieval_intents",
    "evidence_kinds",
)
_TEXT_EVIDENCE_TERMS = {
    "addendum",
    "brief",
    "contract",
    "document",
    "guide",
    "handbook",
    "memo",
    "note",
    "notes",
    "playbook",
    "policy",
    "postmortem",
    "report",
    "review",
    "sla",
    "status",
}
_TABLE_EVIDENCE_TERMS = {
    "alert",
    "alerts",
    "asset",
    "assets",
    "control",
    "controls",
    "csv",
    "deployment",
    "deployments",
    "experiment",
    "experiments",
    "export",
    "exports",
    "feature flag",
    "feature flags",
    "incident",
    "incidents",
    "invoice",
    "invoices",
    "matrix",
    "record",
    "records",
    "rollout",
    "rollouts",
    "row",
    "rows",
    "service",
    "services",
    "table",
    "vendor",
    "vendors",
}
_CODE_EVIDENCE_TERMS = {
    "code",
    "constant",
    "environment variable",
    "env var",
    "function",
    "helper",
    "implementation",
    "method",
    "module",
    "path",
    "python",
    "scheduler",
    "symbol",
}
_TEMPORAL_TERMS = {"current", "final", "fresh", "latest", "newest", "recent", "updated"}


@dataclass(frozen=True)
class RetrievalProfile:
    """Pyrrho pre-retrieval plan plus mechanical executor inputs."""

    # Strategy routing
    strategy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "code": 0.25,
            "section": 0.25,
            "table": 0.15,
            "chunk": 0.35,
        }
    )
    entities: tuple[str, ...] = ()

    # Fetch parameters
    top_k: int = 50
    top_read: int = 50

    # Query expansion (from detection)
    query_variations: list[str] = field(default_factory=list)
    comparison_queries: list[str] = field(default_factory=list)
    comparison_entities: list[str] = field(default_factory=list)

    # Semantic keyword expansion (from the query-prep bus)
    keywords: list[str] = field(default_factory=list)

    # Pyrrho pre-retrieval query signals
    query_contract: str | None = None
    query_contract_confidence: float | None = None
    query_contract_probabilities: dict[str, float] = field(default_factory=dict)
    query_route: str | None = None
    query_route_confidence: float | None = None
    query_route_probabilities: dict[str, float] = field(default_factory=dict)
    answerability_shape: str | None = None
    answerability_shape_confidence: float | None = None
    answerability_shape_probabilities: dict[str, float] = field(default_factory=dict)
    retrieval_modality: str | None = None
    retrieval_modality_confidence: float | None = None
    retrieval_modality_probabilities: dict[str, float] = field(default_factory=dict)
    retrieval_obligation: str | None = None
    retrieval_obligation_confidence: float | None = None
    retrieval_obligation_probabilities: dict[str, float] = field(default_factory=dict)
    retrieval_intents: tuple[str, ...] = ()
    retrieval_intent_probabilities: dict[str, float] = field(default_factory=dict)
    evidence_kinds: tuple[str, ...] = ()
    evidence_kind_probabilities: dict[str, float] = field(default_factory=dict)
    required_modalities: tuple[str, ...] = ()

    # Feature gates
    run_agentic: bool = True
    inject_corpus_summaries: bool = False

    # Temporal metadata (for tagging query variations with references)
    temporal_references: list[str] = field(default_factory=list)

    # Boost signals
    boost_recency: bool = False
    has_aggregation_intent: bool = False
    has_comparison_intent: bool = False
    has_temporal_intent: bool = False

    # Context assembly
    entity_expansion_limit: int = 3

    # Extended advisory signals
    specificity: str = "moderate"
    answer_type: str = "factual"
    domain: str = "general"

    # Source refs (for logging/debugging)
    planning_owner: str = "pyrrho"
    auxiliary_signal_policy: str = "pyrrho_gated"
    analysis_type: str = "general"
    analysis_confidence: float = 0.5


def build_retrieval_profile(
    analysis: "QueryAnalysis | None",
    detection: Any,
    config: "FitzKragConfig",
    *,
    extended_signals: dict[str, Any] | None = None,
    keywords: list[str] | None = None,
    query_signals: Any = None,
) -> RetrievalProfile:
    """Build a Pyrrho-owned RetrievalProfile.

    fitz-sage may provide literal query variants, keywords, and candidate
    fetching mechanics, but semantic planning belongs to Pyrrho. Analysis,
    detection, and extended LLM signals are recorded as auxiliary context only;
    they do not choose route, modality, temporal/comparison/aggregation intent,
    or evidence obligations.
    """
    specificity = "moderate"
    answer_type = "factual"
    domain = "general"
    query_contract = _query_signal_head(query_signals, "query_contract")
    query_route = _query_signal_head(query_signals, "route")
    answerability_shape = _query_signal_head(query_signals, "answerability_shape")
    retrieval_modality = _query_signal_head(query_signals, "retrieval_modality")
    retrieval_obligation = _query_signal_head(query_signals, "retrieval_obligation")
    retrieval_intents_head = _query_signal_head(query_signals, "retrieval_intents")
    evidence_kinds_head = _query_signal_head(query_signals, "evidence_kinds")

    contract_label = _head_label(query_contract)
    contract_confidence = _head_confidence(query_contract)
    contract_probabilities = _head_probabilities(query_contract)
    route_label = _head_label(query_route)
    route_confidence = _head_confidence(query_route)
    route_probabilities = _head_probabilities(query_route)
    shape_label = _head_label(answerability_shape)
    shape_confidence = _head_confidence(answerability_shape)
    shape_probabilities = _head_probabilities(answerability_shape)
    modality_label = _head_label(retrieval_modality)
    modality_confidence = _head_confidence(retrieval_modality)
    modality_probabilities = _head_probabilities(retrieval_modality)
    obligation_label = _head_label(retrieval_obligation)
    obligation_confidence = _head_confidence(retrieval_obligation)
    obligation_probabilities = _head_probabilities(retrieval_obligation)
    retrieval_intents = _head_labels(retrieval_intents_head)
    retrieval_intent_probabilities = _head_probabilities(retrieval_intents_head)
    evidence_kinds = _head_labels(evidence_kinds_head)
    evidence_kind_probabilities = _head_probabilities(evidence_kinds_head)
    fallback_active = not _has_query_signal_heads(query_signals)
    fallback = (
        _deterministic_fallback_signals(analysis, detection, extended_signals)
        if fallback_active
        else {}
    )
    contract_label = contract_label or fallback.get("query_contract")
    shape_label = shape_label or fallback.get("answerability_shape")
    modality_label = modality_label or fallback.get("retrieval_modality")
    obligation_label = obligation_label or fallback.get("retrieval_obligation")
    retrieval_intents = _merge_labels(
        retrieval_intents,
        tuple(fallback.get("retrieval_intents", ())),
    )
    evidence_kinds = _merge_labels(
        evidence_kinds,
        tuple(fallback.get("evidence_kinds", ())),
    )
    required_modalities = required_modalities_from_pyrrho(modality_label, obligation_label)
    required_modalities = _merge_modalities(
        required_modalities,
        required_modalities_from_v2_evidence_kinds(evidence_kinds),
        tuple(fallback.get("required_modalities", ())),
    )
    domain = _ROUTE_DOMAINS.get(route_label, domain)
    if fallback_active and extended_signals:
        specificity = str(extended_signals.get("specificity") or specificity)
        answer_type = str(extended_signals.get("answer_type") or answer_type)
        domain = str(extended_signals.get("domain") or domain)

    if contract_label == "representative_overview":
        specificity = "broad"
        answer_type = "exploratory"
    elif contract_label == "exhaustive_coverage":
        specificity = "broad"
        answer_type = "exploratory"
    elif contract_label == "comparison_coverage":
        answer_type = "comparative"
    elif contract_label == "structured_lookup":
        answer_type = "factual"

    if shape_label == "synthesis_answer" and contract_label not in {
        "comparison_coverage",
        "structured_lookup",
    }:
        specificity = "broad"
        answer_type = "exploratory"
    elif shape_label == "set_answer" and contract_label != "structured_lookup":
        specificity = "broad"
        answer_type = "exploratory"
    elif shape_label == "structured_reasoning" and contract_label not in {
        "comparison_coverage",
        "representative_overview",
        "exhaustive_coverage",
    }:
        answer_type = "procedural"

    if "needs_broad_coverage" in retrieval_intents:
        specificity = "broad"
        answer_type = "exploratory"
    elif "needs_comparison_or_set" in retrieval_intents:
        answer_type = "comparative"

    # --- Auxiliary analysis: logged and used for literal/entity expansion only. ---
    if analysis:
        primary_type = analysis.primary_type.value
        confidence = analysis.confidence
        entities = analysis.entities
    else:
        primary_type = "general"
        confidence = 0.5
        entities = ()
    strategy_weights = (
        dict(getattr(analysis, "strategy_weights", {}) or {})
        if fallback_active and analysis
        else {}
    )
    if not strategy_weights:
        strategy_weights = {
            "code": 0.25,
            "section": 0.25,
            "table": 0.15,
            "chunk": 0.35,
        }

    # --- Detection-derived retrieval text is Pyrrho-gated. ---
    query_variations: list[str] = []
    comparison_queries: list[str] = []
    comparison_entities: list[str] = []
    temporal_references: list[str] = []

    if detection:
        if contract_label in {
            "representative_overview",
            "exhaustive_coverage",
            "comparison_coverage",
        }:
            query_variations = list(getattr(detection, "query_variations", []) or [])
        if contract_label == "comparison_coverage":
            comparison_queries = list(getattr(detection, "comparison_queries", []) or [])
            comparison_entities = list(getattr(detection, "comparison_entities", []) or [])
        if contract_label == "temporal_grounding":
            temporal = getattr(detection, "temporal", None)
            if temporal and getattr(temporal, "detected", False):
                try:
                    refs = temporal.metadata.get("references", [])
                    for r in refs:
                        if isinstance(r, dict):
                            temporal_references.append(r.get("text", ""))
                        elif isinstance(r, str):
                            temporal_references.append(r)
                except Exception:
                    pass

    boost_recency = False
    has_aggregation_intent = False
    has_comparison_intent = False
    has_temporal_intent = False
    if contract_label == "comparison_coverage":
        has_comparison_intent = True
    elif contract_label == "temporal_grounding":
        has_temporal_intent = True
        boost_recency = True
    elif contract_label == "exhaustive_coverage":
        has_aggregation_intent = True
    if shape_label == "set_answer":
        has_aggregation_intent = True
    if "needs_comparison_or_set" in retrieval_intents:
        has_comparison_intent = True
    if "needs_temporal_resolution" in retrieval_intents:
        has_temporal_intent = True
        boost_recency = True
    if "needs_broad_coverage" in retrieval_intents:
        has_aggregation_intent = True

    # --- top_k: base * fetch_multiplier * specificity adjustment ---
    top_k = config.top_addresses
    if specificity == "broad":
        top_k = int(top_k * 1.5)
    elif specificity == "narrow":
        top_k = int(top_k * 0.7)
    top_k = max(10, top_k)

    top_read = config.top_read
    if specificity == "broad":
        top_read = int(top_read * 1.3)
    elif specificity == "narrow":
        top_read = max(5, int(top_read * 0.8))

    # Answer type: procedural/comparative need more context sources
    if answer_type == "procedural":
        top_read = int(top_read * 1.3)
    elif answer_type in ("comparative", "exploratory"):
        top_read = int(top_read * 1.2)

    # --- Executor gates: controlled by Pyrrho plan, never by auxiliary analysis. ---
    run_agentic = True

    inject_corpus_summaries = False
    if specificity == "broad" or answer_type == "exploratory":
        inject_corpus_summaries = True

    if contract_label == "structured_lookup":
        strategy_weights = dict(strategy_weights)
        strategy_weights["code"] = max(strategy_weights.get("code", 0.0), 0.25)
        strategy_weights["table"] = max(strategy_weights.get("table", 0.0), 0.35)
        top_k = max(top_k, config.top_addresses)
        top_read = max(top_read, config.top_read)

    apply_retrieval_modality_weights(strategy_weights, modality_label)
    apply_required_modality_weights(strategy_weights, required_modalities)

    # --- Entity expansion limit ---
    entity_expansion_limit = 3
    if specificity == "broad":
        entity_expansion_limit = 12

    return RetrievalProfile(
        strategy_weights=strategy_weights,
        entities=entities,
        top_k=top_k,
        top_read=top_read,
        query_variations=query_variations,
        comparison_queries=comparison_queries,
        comparison_entities=comparison_entities,
        temporal_references=temporal_references,
        keywords=keywords or [],
        query_contract=contract_label,
        query_contract_confidence=contract_confidence,
        query_contract_probabilities=contract_probabilities,
        query_route=route_label,
        query_route_confidence=route_confidence,
        query_route_probabilities=route_probabilities,
        answerability_shape=shape_label,
        answerability_shape_confidence=shape_confidence,
        answerability_shape_probabilities=shape_probabilities,
        retrieval_modality=modality_label,
        retrieval_modality_confidence=modality_confidence,
        retrieval_modality_probabilities=modality_probabilities,
        retrieval_obligation=obligation_label,
        retrieval_obligation_confidence=obligation_confidence,
        retrieval_obligation_probabilities=obligation_probabilities,
        retrieval_intents=retrieval_intents,
        retrieval_intent_probabilities=retrieval_intent_probabilities,
        evidence_kinds=evidence_kinds,
        evidence_kind_probabilities=evidence_kind_probabilities,
        required_modalities=required_modalities,
        run_agentic=run_agentic,
        inject_corpus_summaries=inject_corpus_summaries,
        boost_recency=boost_recency,
        has_aggregation_intent=has_aggregation_intent,
        has_comparison_intent=has_comparison_intent,
        has_temporal_intent=has_temporal_intent,
        entity_expansion_limit=entity_expansion_limit,
        specificity=specificity,
        answer_type=answer_type,
        domain=domain,
        planning_owner="deterministic_fallback" if fallback_active else "pyrrho",
        auxiliary_signal_policy="query_head_fallback" if fallback_active else "pyrrho_gated",
        analysis_type=primary_type,
        analysis_confidence=confidence,
    )


def apply_retrieval_modality_weights(weights: dict[str, float], modality: str | None) -> None:
    """Bias strategy weights toward Pyrrho's preferred retrieval modality."""
    if modality == "structured_table":
        weights["table"] = max(weights.get("table", 0.0), 0.55)
    elif modality == "code":
        weights["code"] = max(weights.get("code", 0.0), 0.60)
    elif modality in {"configuration", "log_trace", "pdf_layout", "unstructured_text"}:
        weights["section"] = max(weights.get("section", 0.0), 0.45)
    elif modality == "mixed":
        weights["code"] = max(weights.get("code", 0.0), 0.30)
        weights["section"] = max(weights.get("section", 0.0), 0.30)
        weights["table"] = max(weights.get("table", 0.0), 0.25)


def apply_required_modality_weights(
    weights: dict[str, float],
    required_modalities: tuple[str, ...],
) -> None:
    """Guarantee eligible strategies for Pyrrho-required evidence kinds."""
    for modality in required_modalities:
        if modality == "table":
            weights["table"] = max(weights.get("table", 0.0), 0.55)
        elif modality == "symbol":
            weights["code"] = max(weights.get("code", 0.0), 0.60)
        elif modality == "section":
            weights["section"] = max(weights.get("section", 0.0), 0.45)


def _deterministic_fallback_signals(
    analysis: Any,
    detection: Any,
    extended_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build recall-only fallback signals when v2 query heads are inactive."""
    query = str(getattr(analysis, "refined_query", "") or "")
    lower = query.lower()
    analysis_type = str(getattr(getattr(analysis, "primary_type", None), "value", "general"))
    kinds: list[str] = []
    intents: list[str] = ["needs_lookup"]

    def add_kind(kind: str) -> None:
        if kind not in kinds:
            kinds.append(kind)

    if analysis_type == "data":
        add_kind("needs_table_or_record")
    elif analysis_type == "code":
        add_kind("needs_code_or_symbol")
    elif analysis_type == "documentation":
        add_kind("needs_text")

    if _contains_any(lower, _TEXT_EVIDENCE_TERMS) or lower.startswith("using "):
        add_kind("needs_text")
    if _contains_any(lower, _TABLE_EVIDENCE_TERMS) or _has_record_identifier(query):
        add_kind("needs_table_or_record")
    if _contains_any(lower, _CODE_EVIDENCE_TERMS) or re.search(
        r"\b[A-Za-z0-9_.-]+\.py\b",
        query,
    ):
        add_kind("needs_code_or_symbol")

    has_comparison = bool(getattr(detection, "has_comparison_intent", False))
    has_temporal = bool(
        getattr(detection, "has_temporal_intent", False)
        or getattr(detection, "has_freshness_intent", False)
        or _contains_any(lower, _TEMPORAL_TERMS)
    )
    has_multi_source_hint = (
        len(kinds) > 1
        or lower.startswith("using ")
        or " according to " in lower
        or " against " in lower
    )
    if has_comparison or has_multi_source_hint:
        intents.append("needs_comparison_or_set")
    if has_temporal:
        intents.append("needs_temporal_resolution")

    query_contract = None
    if has_temporal:
        query_contract = "temporal_grounding"
    elif has_comparison or has_multi_source_hint:
        query_contract = "comparison_coverage"
    elif "needs_table_or_record" in kinds:
        query_contract = "structured_lookup"

    retrieval_modality = None
    retrieval_obligation = None
    if len(kinds) > 1:
        retrieval_modality = "mixed"
        if "needs_text" in kinds and "needs_table_or_record" in kinds:
            retrieval_obligation = "prose_plus_table"
        if "needs_text" in kinds and "needs_code_or_symbol" in kinds:
            retrieval_obligation = retrieval_obligation or "prose_plus_code"
    elif kinds == ["needs_table_or_record"]:
        retrieval_modality = "structured_table"
    elif kinds == ["needs_code_or_symbol"]:
        retrieval_modality = "code"
    elif kinds == ["needs_text"]:
        retrieval_modality = "unstructured_text"

    answerability_shape = None
    if has_comparison or has_multi_source_hint:
        answerability_shape = "structured_reasoning"

    return {
        "query_contract": query_contract,
        "answerability_shape": answerability_shape,
        "retrieval_modality": retrieval_modality,
        "retrieval_obligation": retrieval_obligation,
        "retrieval_intents": tuple(dict.fromkeys(intents)),
        "evidence_kinds": tuple(kinds),
        "required_modalities": required_modalities_from_v2_evidence_kinds(kinds),
    }


def _has_query_signal_heads(query_signals: Any) -> bool:
    """Return whether Pyrrho supplied an active query-side head."""
    return any(
        _head_labels(_query_signal_head(query_signals, name))
        for name in _HEAD_NAMES
    )


def _merge_labels(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable unique labels."""
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return tuple(merged)


def _contains_any(lower_text: str, terms: set[str]) -> bool:
    """Return whether a lower-cased query contains any whole-word term."""
    return any(re.search(rf"\b{re.escape(term)}\b", lower_text) for term in terms)


def _has_record_identifier(query: str) -> bool:
    """Return whether query includes a table-like id such as EXP-505 or ROL-401."""
    return bool(re.search(r"\b[A-Z]{2,}\d*[A-Z]*-\d+\b|\b[A-Z]{2,}\d+\b", query))


def query_profile_metadata(query_signals: Any, profile: RetrievalProfile | None) -> dict[str, Any]:
    """Build serializable metadata for Pyrrho's pre-retrieval query plan."""
    if profile is None:
        return {}
    signals = {
        "query_contract": _query_signal_metadata(
            _query_signal_head(query_signals, "query_contract"),
            applied_label=profile.query_contract,
        ),
        "route": _query_signal_metadata(
            _query_signal_head(query_signals, "route"),
            applied_label=profile.query_route,
        ),
        "answerability_shape": _query_signal_metadata(
            _query_signal_head(query_signals, "answerability_shape"),
            applied_label=profile.answerability_shape,
        ),
        "retrieval_modality": _query_signal_metadata(
            _query_signal_head(query_signals, "retrieval_modality"),
            applied_label=profile.retrieval_modality,
        ),
        "retrieval_obligation": _query_signal_metadata(
            _query_signal_head(query_signals, "retrieval_obligation"),
            applied_label=profile.retrieval_obligation,
        ),
        "retrieval_intents": _query_signal_metadata(
            _query_signal_head(query_signals, "retrieval_intents"),
            applied_label=None,
        ),
        "evidence_kinds": _query_signal_metadata(
            _query_signal_head(query_signals, "evidence_kinds"),
            applied_label=None,
        ),
    }
    return {
        "signals": {key: value for key, value in signals.items() if value},
        "profile": _profile_metadata(profile),
    }


def _query_signal_metadata(head: Any, *, applied_label: str | None) -> dict[str, Any]:
    """Serialize one Pyrrho query head with its retrieval-use decision."""
    label = _head_label(head)
    labels = _head_labels(head)
    if label is None and not labels:
        return {}
    metadata = {
        "final_label": label,
        "confidence": _head_confidence(head),
        "used_for_retrieval": bool(labels) if applied_label is None else applied_label in labels,
    }
    if labels:
        metadata["final_labels"] = list(labels)
    raw_label = getattr(head, "raw_label", None)
    if raw_label:
        metadata["raw_label"] = str(raw_label)
    probabilities = _head_probabilities(head)
    if probabilities:
        metadata["probabilities"] = probabilities
    return metadata


def _profile_metadata(profile: RetrievalProfile) -> dict[str, Any]:
    """Serialize retrieval-profile knobs that materially affect recall."""
    return {
        "query_contract": profile.query_contract,
        "query_route": profile.query_route,
        "answerability_shape": profile.answerability_shape,
        "retrieval_modality": profile.retrieval_modality,
        "retrieval_obligation": profile.retrieval_obligation,
        "retrieval_intents": list(profile.retrieval_intents),
        "evidence_kinds": list(profile.evidence_kinds),
        "required_modalities": list(profile.required_modalities),
        "domain": profile.domain,
        "specificity": profile.specificity,
        "answer_type": profile.answer_type,
        "top_k": profile.top_k,
        "top_read": profile.top_read,
        "strategy_weights": {
            key: float(value) for key, value in sorted(profile.strategy_weights.items())
        },
        "run_agentic": profile.run_agentic,
        "inject_corpus_summaries": profile.inject_corpus_summaries,
        "boost_recency": profile.boost_recency,
        "has_aggregation_intent": profile.has_aggregation_intent,
        "has_comparison_intent": profile.has_comparison_intent,
        "has_temporal_intent": profile.has_temporal_intent,
        "entity_expansion_limit": profile.entity_expansion_limit,
        "planning_owner": profile.planning_owner,
        "auxiliary_signal_policy": profile.auxiliary_signal_policy,
    }


def _query_signal_head(query_signals: Any, name: str) -> Any:
    """Return one head from a Pyrrho query decision."""
    head = getattr(query_signals, name, None)
    if head is not None:
        return head
    heads = getattr(query_signals, "heads", None)
    if isinstance(heads, dict):
        return heads.get(name)
    return None


def _head_label(head: Any) -> str | None:
    """Return Pyrrho's final head label if available."""
    label = getattr(head, "final_label", None)
    return str(label) if label else None


def _head_labels(head: Any) -> tuple[str, ...]:
    """Return Pyrrho's final labels for single- or multi-label heads."""
    labels = getattr(head, "final_labels", None)
    if labels is not None and not isinstance(labels, str):
        return tuple(str(label) for label in labels if label)
    label = _head_label(head)
    return (label,) if label else ()


def _head_confidence(head: Any) -> float | None:
    """Return Pyrrho head confidence if available."""
    confidence = getattr(head, "confidence", None)
    return float(confidence) if isinstance(confidence, int | float) else None


def _head_probabilities(head: Any) -> dict[str, float]:
    """Return serializable head probabilities."""
    probabilities = getattr(head, "probabilities", None)
    if not isinstance(probabilities, dict):
        return {}
    return {str(key): float(value) for key, value in probabilities.items()}


def _merge_modalities(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable unique retrieval modality requirements."""
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return tuple(merged)
