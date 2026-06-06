# fitz_sage/engines/fitz_krag/retrieval_profile.py
"""
Unified retrieval profile — built once from analysis + detection + config.

Replaces scattered gating logic (router._should_run_*, engine._is_thematic,
config reads) with a single object that all retrieval consumers read from.

Extended signals (specificity, domain, etc.) are ADVISORY — soft multipliers
on defaults. Missing signals = current behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis

_QUERY_SIGNAL_MIN_CONFIDENCE = {
    "query_contract": 0.55,
    "route": 0.75,
    "answerability_shape": 0.70,
    "retrieval_modality": 0.60,
}

_ROUTE_DOMAINS = {
    "technology_computing": "technical",
    "law_policy": "legal",
    "science_medicine": "medical",
    "economics_finance": "financial",
    "history_geography": "general",
    "culture_society": "general",
    "general_commonsense": "general",
}


@dataclass(frozen=True)
class RetrievalProfile:
    """Unified retrieval tuning — all consumers read from this."""

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
    """Build a RetrievalProfile from classification outputs + config.

    Pure function — no side effects. Absorbs all gating logic previously
    scattered across router._should_run_*, engine._is_thematic, etc.

    Args:
        analysis: Query type classification (None = no signal).
        detection: DetectionSummary (None = no detection ran).
        config: Engine config with default values.
        extended_signals: Optional dict from LLM with advisory signals.
        keywords: Semantic keyword expansion from the query-prep bus.
    """
    ext = extended_signals or {}
    specificity = ext.get("specificity", "moderate")
    answer_type = ext.get("answer_type", "factual")
    domain = ext.get("domain", "general")
    query_contract = _query_signal_head(query_signals, "query_contract")
    query_route = _query_signal_head(query_signals, "route")
    answerability_shape = _query_signal_head(query_signals, "answerability_shape")
    retrieval_modality = _query_signal_head(query_signals, "retrieval_modality")

    contract_label = _trusted_head_label(query_contract, "query_contract")
    contract_confidence = _head_confidence(query_contract)
    contract_probabilities = _head_probabilities(query_contract)
    route_label = _trusted_head_label(query_route, "route")
    route_confidence = _head_confidence(query_route)
    route_probabilities = _head_probabilities(query_route)
    shape_label = _trusted_head_label(answerability_shape, "answerability_shape")
    shape_confidence = _head_confidence(answerability_shape)
    shape_probabilities = _head_probabilities(answerability_shape)
    modality_label = _trusted_head_label(retrieval_modality, "retrieval_modality")
    modality_confidence = _head_confidence(retrieval_modality)
    modality_probabilities = _head_probabilities(retrieval_modality)
    domain = _ROUTE_DOMAINS.get(route_label, domain)

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

    # --- Analysis-derived ---
    if analysis:
        primary_type = analysis.primary_type.value
        confidence = analysis.confidence
        strategy_weights = dict(analysis.strategy_weights)
        entities = analysis.entities
    else:
        primary_type = "general"
        confidence = 0.5
        from fitz_sage.engines.fitz_krag.query_analyzer import _TYPE_WEIGHTS, QueryType

        strategy_weights = dict(_TYPE_WEIGHTS[QueryType.GENERAL])
        entities = ()

    # --- Detection-derived ---
    fetch_multiplier = 1
    query_variations: list[str] = []
    comparison_queries: list[str] = []
    comparison_entities: list[str] = []
    boost_recency = False
    has_aggregation_intent = False
    has_comparison_intent = False
    has_temporal_intent = False

    temporal_references: list[str] = []

    if detection:
        has_aggregation_intent = bool(getattr(detection, "has_aggregation_intent", False))
        has_comparison_intent = bool(getattr(detection, "has_comparison_intent", False))
        has_temporal_intent = bool(getattr(detection, "has_temporal_intent", False))
        fetch_multiplier = getattr(detection, "fetch_multiplier", 1) or 1
        query_variations = list(getattr(detection, "query_variations", []) or [])
        comparison_queries = list(getattr(detection, "comparison_queries", []) or [])
        comparison_entities = list(getattr(detection, "comparison_entities", []) or [])
        boost_recency = bool(getattr(detection, "boost_recency", False))

        # Extract temporal references for tagging query variations
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

    if contract_label == "comparison_coverage":
        has_comparison_intent = True
    elif contract_label == "temporal_grounding":
        has_temporal_intent = True
        boost_recency = True
    elif contract_label == "exhaustive_coverage":
        has_aggregation_intent = True
    if shape_label == "set_answer":
        has_aggregation_intent = True

    # --- top_k: base * fetch_multiplier * specificity adjustment ---
    top_k = config.top_addresses * fetch_multiplier
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

    # --- Agentic gate (from router._should_run_agentic) ---
    run_agentic = True
    if analysis and primary_type == "data":
        run_agentic = False

    # --- Corpus summaries gate ---
    # L2 corpus summaries answer overview queries — gate on broad/exploratory intent.
    inject_corpus_summaries = False
    if analysis and primary_type not in ("code", "data"):
        if specificity == "broad" or answer_type == "exploratory":
            inject_corpus_summaries = True

    if contract_label == "structured_lookup":
        strategy_weights = dict(strategy_weights)
        strategy_weights["table"] = max(strategy_weights.get("table", 0.0), 0.35)
        top_k = max(top_k, config.top_addresses)
        top_read = max(top_read, config.top_read)

    apply_retrieval_modality_weights(strategy_weights, modality_label)

    # --- Entity expansion limit (from engine._is_thematic) ---
    is_thematic = analysis is not None and primary_type not in ("code", "data") and confidence < 0.6
    entity_expansion_limit = 12 if is_thematic else 3
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
    }
    return {
        "signals": {key: value for key, value in signals.items() if value},
        "profile": _profile_metadata(profile),
    }


def _query_signal_metadata(head: Any, *, applied_label: str | None) -> dict[str, Any]:
    """Serialize one Pyrrho query head with its retrieval-use decision."""
    label = _head_label(head)
    if label is None:
        return {}
    metadata = {
        "final_label": label,
        "confidence": _head_confidence(head),
        "used_for_retrieval": applied_label == label,
    }
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


def _trusted_head_label(head: Any, name: str) -> str | None:
    """Return a label only when the head is confident enough to steer retrieval."""
    label = _head_label(head)
    confidence = _head_confidence(head)
    if label is None or confidence is None:
        return None
    threshold = _QUERY_SIGNAL_MIN_CONFIDENCE.get(name, 1.0)
    return label if confidence >= threshold else None


def _head_label(head: Any) -> str | None:
    """Return Pyrrho's final head label if available."""
    label = getattr(head, "final_label", None)
    return str(label) if label else None


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
