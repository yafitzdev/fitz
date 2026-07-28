# fitz_sage/engines/fitz_krag/retrieval_profile.py
"""Retrieval profile construction for the KRAG executor."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.evidence_contract import (
    required_modalities_from_profile,
    required_modalities_from_v2_evidence_kinds,
)

if TYPE_CHECKING:
    from pyrrho import MultiLabelDecision, QueryPlan as PyrrhoQueryPlan

    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis

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
_TEMPORAL_TERMS = {
    "at that time",
    "current",
    "currently",
    "effective now",
    "final",
    "formerly",
    "fresh",
    "latest",
    "newest",
    "now",
    "original",
    "previous",
    "prior",
    "recent",
    "superseded",
    "updated",
}


@dataclass(frozen=True)
class RetrievalProfile:
    """Retrieval plan plus mechanical executor inputs."""

    # Strategy routing
    strategy_weights: dict[str, float] = field(
        default_factory=lambda: {
            "code": 0.38,
            "section": 0.39,
            "table": 0.23,
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

    # Deterministic retrieval-profile signals
    query_contract: str | None = None
    answerability_shape: str | None = None
    retrieval_modality: str | None = None
    retrieval_obligation: str | None = None
    retrieval_intents: tuple[str, ...] = ()
    evidence_kinds: tuple[str, ...] = ()
    required_modalities: tuple[str, ...] = ()

    # Feature gates
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
    planning_owner: str = "fitz_krag"
    auxiliary_signal_policy: str = "deterministic_profile"
    analysis_type: str = "general"
    analysis_confidence: float = 0.5


def build_retrieval_profile(
    analysis: "QueryAnalysis | None",
    detection: Any,
    config: "FitzKragConfig",
    *,
    extended_signals: dict[str, Any] | None = None,
    keywords: list[str] | None = None,
    pyrrho_plan: "PyrrhoQueryPlan | None" = None,
) -> RetrievalProfile:
    """Build a retrieval profile for the KRAG executor.

    Pyrrho's query-only v2 heads contribute evidence-intent signals when
    available. Fitz-owned deterministic query-shape signals remain active so
    explicit temporal, comparison, and aggregation language cannot be dropped.
    """
    specificity = "moderate"
    answer_type = "factual"
    domain = "general"
    fallback = _deterministic_profile_signals(analysis, detection, extended_signals)
    pyrrho_signals = _pyrrho_profile_signals(pyrrho_plan)
    planning_owner = "fitz_krag"
    auxiliary_signal_policy = "deterministic_profile"
    if pyrrho_signals:
        fallback = _merge_pyrrho_signals(fallback, pyrrho_signals)
        planning_owner = "hybrid"
        auxiliary_signal_policy = "pyrrho_v2_pre_plus_deterministic_query_shape"
    contract_label = fallback.get("query_contract")
    shape_label = fallback.get("answerability_shape")
    modality_label = fallback.get("retrieval_modality")
    obligation_label = fallback.get("retrieval_obligation")
    retrieval_intents = tuple(fallback.get("retrieval_intents", ()))
    evidence_kinds = tuple(fallback.get("evidence_kinds", ()))
    required_modalities = required_modalities_from_profile(modality_label, obligation_label)
    required_modalities = _merge_modalities(
        required_modalities,
        required_modalities_from_v2_evidence_kinds(evidence_kinds),
        tuple(fallback.get("required_modalities", ())),
    )
    if extended_signals:
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
    strategy_weights = dict(getattr(analysis, "strategy_weights", {}) or {}) if analysis else {}
    if not strategy_weights:
        strategy_weights = {
            "code": 0.38,
            "section": 0.39,
            "table": 0.23,
        }

    # --- Detection-derived retrieval text is query-head gated. ---
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

    deterministic_aggregation = bool(getattr(detection, "has_aggregation_intent", False))
    deterministic_comparison = bool(getattr(detection, "has_comparison_intent", False))
    deterministic_temporal = bool(
        getattr(detection, "has_temporal_intent", False)
        or getattr(detection, "has_freshness_intent", False)
        or _contains_any(
            str(getattr(analysis, "refined_query", "") or "").lower(),
            _TEMPORAL_TERMS,
        )
    )
    # Query-shape fields describe the user's request. Pyrrho PRE heads describe
    # evidence obligations and remain available in retrieval_intents/metadata;
    # they must not relabel a narrow query as comparison, aggregation, or time.
    has_aggregation_intent = deterministic_aggregation
    has_comparison_intent = deterministic_comparison
    has_temporal_intent = deterministic_temporal
    boost_recency = has_temporal_intent

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
        answerability_shape=shape_label,
        retrieval_modality=modality_label,
        retrieval_obligation=obligation_label,
        retrieval_intents=retrieval_intents,
        evidence_kinds=evidence_kinds,
        required_modalities=required_modalities,
        inject_corpus_summaries=inject_corpus_summaries,
        boost_recency=boost_recency,
        has_aggregation_intent=has_aggregation_intent,
        has_comparison_intent=has_comparison_intent,
        has_temporal_intent=has_temporal_intent,
        entity_expansion_limit=entity_expansion_limit,
        specificity=specificity,
        answer_type=answer_type,
        domain=domain,
        planning_owner=planning_owner,
        auxiliary_signal_policy=auxiliary_signal_policy,
        analysis_type=primary_type,
        analysis_confidence=confidence,
    )


def apply_retrieval_modality_weights(weights: dict[str, float], modality: str | None) -> None:
    """Bias strategy weights toward the profile's preferred retrieval modality."""
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
    """Guarantee eligible strategies for profile-required evidence kinds."""
    for modality in required_modalities:
        if modality == "table":
            weights["table"] = max(weights.get("table", 0.0), 0.55)
        elif modality == "symbol":
            weights["code"] = max(weights.get("code", 0.0), 0.60)
        elif modality == "section":
            weights["section"] = max(weights.get("section", 0.0), 0.45)


def _deterministic_profile_signals(
    analysis: Any,
    detection: Any,
    extended_signals: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build recall-only profile signals from query text and detection."""
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
    has_aggregation = bool(getattr(detection, "has_aggregation_intent", False))
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
    if has_aggregation:
        intents.append("needs_broad_coverage")
    if has_temporal:
        intents.append("needs_temporal_resolution")

    query_contract = None
    if has_temporal:
        query_contract = "temporal_grounding"
    elif has_aggregation:
        query_contract = "exhaustive_coverage"
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
    if has_aggregation:
        answerability_shape = "synthesis_answer"
    elif has_comparison or has_multi_source_hint:
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


def _pyrrho_profile_signals(pyrrho_plan: "PyrrhoQueryPlan | None") -> dict[str, Any]:
    """Translate Pyrrho PRE heads into retrieval-profile signals."""
    if pyrrho_plan is None:
        return {}
    intents = _head_final_labels(pyrrho_plan.retrieval_intents)
    kinds = _head_final_labels(pyrrho_plan.evidence_kinds)
    if not intents and not kinds:
        return {}

    query_contract = None
    answerability_shape = None
    if "needs_temporal_resolution" in intents:
        query_contract = "temporal_grounding"
    elif "needs_broad_coverage" in intents:
        query_contract = "exhaustive_coverage"
        answerability_shape = "synthesis_answer"
    elif "needs_comparison_or_set" in intents:
        query_contract = "comparison_coverage"
        answerability_shape = "structured_reasoning"
    elif "needs_table_or_record" in kinds:
        query_contract = "structured_lookup"

    retrieval_modality, retrieval_obligation = _modality_from_evidence_kinds(kinds)
    return {
        "query_contract": query_contract,
        "answerability_shape": answerability_shape,
        "retrieval_modality": retrieval_modality,
        "retrieval_obligation": retrieval_obligation,
        "retrieval_intents": intents,
        "evidence_kinds": kinds,
        "required_modalities": required_modalities_from_v2_evidence_kinds(kinds),
    }


def _merge_pyrrho_signals(
    fallback: dict[str, Any],
    pyrrho_signals: dict[str, Any],
) -> dict[str, Any]:
    """Union model query heads with deterministic recall signals."""
    merged = dict(fallback)
    for key in ("retrieval_intents", "evidence_kinds", "required_modalities"):
        merged[key] = tuple(
            dict.fromkeys(
                (
                    *tuple(fallback.get(key) or ()),
                    *tuple(pyrrho_signals.get(key) or ()),
                )
            )
        )

    intents = tuple(merged["retrieval_intents"])
    kinds = tuple(merged["evidence_kinds"])
    if "needs_temporal_resolution" in intents:
        merged["query_contract"] = "temporal_grounding"
        merged["answerability_shape"] = None
    elif "needs_broad_coverage" in intents:
        merged["query_contract"] = "exhaustive_coverage"
        merged["answerability_shape"] = "synthesis_answer"
    elif "needs_comparison_or_set" in intents:
        merged["query_contract"] = "comparison_coverage"
        merged["answerability_shape"] = "structured_reasoning"
    elif "needs_table_or_record" in kinds:
        merged["query_contract"] = "structured_lookup"

    modality, obligation = _modality_from_evidence_kinds(kinds)
    if modality:
        merged["retrieval_modality"] = modality
        merged["retrieval_obligation"] = obligation
    return merged


def _head_final_labels(head: "MultiLabelDecision") -> tuple[str, ...]:
    """Return Pyrrho's final labels without reinterpreting them."""
    return tuple(dict.fromkeys(head.final_labels))


def _modality_from_evidence_kinds(kinds: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Map v2 evidence-kind labels onto retrieval modality knobs."""
    if len(kinds) > 1:
        retrieval_obligation = None
        if "needs_text" in kinds and "needs_table_or_record" in kinds:
            retrieval_obligation = "prose_plus_table"
        if "needs_text" in kinds and "needs_code_or_symbol" in kinds:
            retrieval_obligation = retrieval_obligation or "prose_plus_code"
        return "mixed", retrieval_obligation
    if kinds == ("needs_table_or_record",):
        return "structured_table", None
    if kinds == ("needs_code_or_symbol",):
        return "code", None
    if kinds == ("needs_config_or_setting",):
        return "configuration", None
    if kinds == ("needs_log_or_run_result",):
        return "log_trace", None
    if kinds == ("needs_document_layout",):
        return "pdf_layout", None
    if kinds == ("needs_text",):
        return "unstructured_text", None
    return None, None


def _contains_any(lower_text: str, terms: set[str]) -> bool:
    """Return whether a lower-cased query contains any whole-word term."""
    return any(re.search(rf"\b{re.escape(term)}\b", lower_text) for term in terms)


def _has_record_identifier(query: str) -> bool:
    """Return whether query includes a table-like identifier such as ABC-123."""
    return bool(re.search(r"\b[A-Z]{2,}\d*[A-Z]*-\d+\b|\b[A-Z]{2,}\d+\b", query))


def query_profile_metadata(
    profile: RetrievalProfile | None,
    pyrrho_plan: "PyrrhoQueryPlan | None" = None,
) -> dict[str, Any]:
    """Build serializable metadata for the pre-retrieval query profile."""
    if profile is None:
        return {}
    metadata = {"profile": _profile_metadata(profile)}
    pre_metadata = _pyrrho_plan_metadata(pyrrho_plan)
    if pre_metadata:
        metadata["pyrrho_pre"] = pre_metadata
    return metadata


def _pyrrho_plan_metadata(pyrrho_plan: "PyrrhoQueryPlan | None") -> dict[str, Any]:
    """Serialize Pyrrho PRE heads for evidence-pack metadata."""
    if pyrrho_plan is None:
        return {}
    output: dict[str, Any] = {}
    for name, head in (
        ("retrieval_intents", pyrrho_plan.retrieval_intents),
        ("evidence_kinds", pyrrho_plan.evidence_kinds),
    ):
        final_labels = _head_final_labels(head)
        output[name] = {
            "final_labels": list(final_labels),
            "final_label": head.final_label,
            "confidence": float(head.confidence),
            "probabilities": {
                str(label): float(value)
                for label, value in head.probabilities.items()
            },
        }
    return output


def _profile_metadata(profile: RetrievalProfile) -> dict[str, Any]:
    """Serialize retrieval-profile knobs that materially affect recall."""
    return {
        "query_contract": profile.query_contract,
        "answerability_shape": profile.answerability_shape,
        "retrieval_modality": profile.retrieval_modality,
        "retrieval_obligation": profile.retrieval_obligation,
        "retrieval_intents": list(profile.retrieval_intents),
        "evidence_kinds": list(profile.evidence_kinds),
        "required_modalities": list(profile.required_modalities),
        "keywords": list(profile.keywords),
        "query_variations": list(profile.query_variations),
        "comparison_queries": list(profile.comparison_queries),
        "comparison_entities": list(profile.comparison_entities),
        "temporal_references": list(profile.temporal_references),
        "domain": profile.domain,
        "specificity": profile.specificity,
        "answer_type": profile.answer_type,
        "top_k": profile.top_k,
        "top_read": profile.top_read,
        "strategy_weights": {
            key: float(value) for key, value in sorted(profile.strategy_weights.items())
        },
        "inject_corpus_summaries": profile.inject_corpus_summaries,
        "boost_recency": profile.boost_recency,
        "has_aggregation_intent": profile.has_aggregation_intent,
        "has_comparison_intent": profile.has_comparison_intent,
        "has_temporal_intent": profile.has_temporal_intent,
        "entity_expansion_limit": profile.entity_expansion_limit,
        "planning_owner": profile.planning_owner,
        "auxiliary_signal_policy": profile.auxiliary_signal_policy,
    }


def _merge_modalities(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Return stable unique retrieval modality requirements."""
    merged: list[str] = []
    for group in groups:
        for item in group:
            if item not in merged:
                merged.append(item)
    return tuple(merged)
