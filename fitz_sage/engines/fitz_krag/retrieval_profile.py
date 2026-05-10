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

    # Feature gates
    run_hyde: bool = True
    run_multi_query: bool = False
    run_agentic: bool = True
    inject_corpus_summaries: bool = False
    fallback_to_chunks: bool = True

    # Temporal metadata (for tagging query variations with references)
    temporal_references: list[str] = field(default_factory=list)

    # Boost signals
    boost_recency: bool = False
    boost_authority: bool = False

    # Context assembly
    entity_expansion_limit: int = 3

    # Extended advisory signals
    specificity: str = "moderate"
    answer_type: str = "factual"
    domain: str = "general"
    multi_hop: bool = False

    # Source refs (for logging/debugging)
    analysis_type: str = "general"
    analysis_confidence: float = 0.5


def build_retrieval_profile(
    analysis: "QueryAnalysis | None",
    detection: Any,
    config: "FitzKragConfig",
    *,
    query_length: int = 0,
    has_chat_factory: bool = False,
    extended_signals: dict[str, Any] | None = None,
) -> RetrievalProfile:
    """Build a RetrievalProfile from classification outputs + config.

    Pure function — no side effects. Absorbs all gating logic previously
    scattered across router._should_run_*, engine._is_thematic, etc.

    Args:
        analysis: Query type classification (None = no signal).
        detection: DetectionSummary (None = no detection ran).
        config: Engine config with default values.
        query_length: Length of retrieval query (for multi-query gate).
        has_chat_factory: Whether chat factory is available (for multi-query).
        extended_signals: Optional dict from LLM with advisory signals.
    """
    ext = extended_signals or {}
    specificity = ext.get("specificity", "moderate")
    answer_type = ext.get("answer_type", "factual")
    domain = ext.get("domain", "general")
    multi_hop = ext.get("multi_hop", False)

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
    boost_authority = False
    has_complex_intent = False

    temporal_references: list[str] = []

    if detection:
        fetch_multiplier = getattr(detection, "fetch_multiplier", 1) or 1
        query_variations = list(getattr(detection, "query_variations", []) or [])
        comparison_queries = list(getattr(detection, "comparison_queries", []) or [])
        comparison_entities = list(getattr(detection, "comparison_entities", []) or [])
        boost_recency = bool(getattr(detection, "boost_recency", False))
        boost_authority = bool(getattr(detection, "boost_authority", False))
        has_complex_intent = bool(
            getattr(detection, "has_comparison_intent", False)
            or getattr(detection, "has_temporal_intent", False)
            or getattr(detection, "has_aggregation_intent", False)
        )

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

    # HyDE removed: no embedder, no hypothesis-document leg.
    run_hyde = False

    # --- Multi-query gate (from router._should_run_multi_query) ---
    run_multi_query = (
        config.enable_multi_query
        and has_chat_factory
        and query_length >= config.multi_query_min_length
    )
    if run_multi_query and analysis:
        if primary_type in ("code", "data"):
            run_multi_query = False
        elif confidence >= 0.8:
            run_multi_query = False

    # --- Agentic gate (from router._should_run_agentic) ---
    run_agentic = True
    if analysis and primary_type == "data":
        run_agentic = False

    # --- Corpus summaries gate (from router._should_inject_corpus_summaries) ---
    inject_corpus_summaries = False
    if analysis and primary_type not in ("code", "data") and confidence < 0.6:
        inject_corpus_summaries = True

    # ChunkFallbackStrategy deleted; gate kept on the profile for caller compat.
    fallback_to_chunks = False

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
        run_hyde=run_hyde,
        run_multi_query=run_multi_query,
        run_agentic=run_agentic,
        inject_corpus_summaries=inject_corpus_summaries,
        fallback_to_chunks=fallback_to_chunks,
        boost_recency=boost_recency,
        boost_authority=boost_authority,
        entity_expansion_limit=entity_expansion_limit,
        specificity=specificity,
        answer_type=answer_type,
        domain=domain,
        multi_hop=multi_hop,
        analysis_type=primary_type,
        analysis_confidence=confidence,
    )
