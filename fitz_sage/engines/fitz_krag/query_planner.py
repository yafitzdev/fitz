# fitz_sage/engines/fitz_krag/query_planner.py
"""Deterministic query planning for retrieval-first KRAG."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
from fitz_sage.retrieval.detection.modules import (
    AggregationModule,
    AggregationType,
    ComparisonModule,
    FreshnessModule,
    TemporalIntent,
    TemporalModule,
)
from fitz_sage.retrieval.detection.registry import DetectionSummary
from fitz_sage.retrieval.rewriter.types import RewriteResult, RewriteType

_STOP_WORDS = {
    "what",
    "which",
    "where",
    "when",
    "why",
    "how",
    "does",
    "did",
    "the",
    "this",
    "that",
    "with",
    "from",
    "have",
    "has",
    "are",
    "was",
    "were",
    "been",
    "being",
    "will",
    "would",
    "could",
    "should",
    "about",
    "there",
    "their",
    "they",
    "them",
    "than",
    "then",
    "into",
    "also",
    "just",
    "only",
    "very",
    "some",
    "more",
    "most",
    "each",
    "other",
    "your",
    "our",
    "can",
    "not",
    "for",
    "and",
    "but",
}

_CODE_TERMS = {
    "function",
    "class",
    "method",
    "implementation",
    "module",
    "api",
    "code",
    "stacktrace",
    "endpoint",
    "service",
}
_DATA_TERMS = {"csv", "table", "spreadsheet", "row", "column", "sql"}
_DOC_TERMS = {"section", "document", "policy", "procedure", "manual", "spec"}
_COMPARATIVE_PATTERN = (
    r"\b(compare|contrast|differ|differs|differences?|difference between|"
    r"different (?:about|from)|"
    r"changed between|changes between|change between|better|worse|higher|lower|greater|"
    r"less|more|fewer|larger|smaller|highest|lowest|best|worst)\b"
)
_MONTH_PATTERN = (
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december)(?:\s+\d{4})?\b"
)
_CLAUSE_HEAD = (
    r"(?:what|which|who|where|when|why|how|"
    r"is|are|was|were|do|does|did|can|could|should|would|"
    r"list|show|find|identify|provide|give|return|tell|explain|describe)\b"
)
_EXPLICIT_CLAUSE_BOUNDARY = re.compile(
    rf"(?:"
    rf"[?;]\s*(?:(?:and|but|also|then)\s+)?|"
    rf"(?:,\s*)?(?:and|but)\s+(?:also\s+)?|"
    rf"\n+\s*(?:[-*]\s*)?"
    rf")(?={_CLAUSE_HEAD})",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryPlan:
    """Query-prep output consumed by retrieval profile construction."""

    retrieval_query: str
    analysis: QueryAnalysis
    detection: DetectionSummary | None = None
    rewrite_result: RewriteResult | None = None
    extended_signals: dict[str, Any] | None = None
    keywords: list[str] = field(default_factory=list)


class DeterministicQueryPlanner:
    """No-chat query planner based on lexical and query-shape signals."""

    def plan(self, query: str, *, detection_enabled: bool = True) -> QueryPlan:
        """Build a retrieval plan without calling a chat model."""
        terms = content_terms(query)
        detection = deterministic_detection(query) if detection_enabled else None
        answer_type = _answer_type(query, detection)

        return QueryPlan(
            retrieval_query=query,
            analysis=deterministic_analysis(query, terms),
            detection=detection,
            rewrite_result=_deterministic_decomposition(query),
            extended_signals={
                "specificity": _specificity(query, detection),
                "answer_type": answer_type,
                "domain": _domain(query, terms),
            },
            keywords=deterministic_keywords(query, terms),
        )


def plan_from_batch_result(
    query: str,
    batch_result: BatchResult,
    *,
    fallback_analysis: QueryAnalysis,
    detection: DetectionSummary | None,
    fallback_plan: QueryPlan | None = None,
) -> QueryPlan:
    """Normalize configured LLM query-prep output into a QueryPlan."""
    retrieval_query = query
    rewrite_result = _merge_rewrite_results(
        batch_result.rewrite_result,
        fallback_plan.rewrite_result if fallback_plan else None,
    )
    if rewrite_result and rewrite_result.rewritten_query != query:
        retrieval_query = rewrite_result.rewritten_query

    return QueryPlan(
        retrieval_query=retrieval_query,
        analysis=batch_result.analysis or fallback_analysis,
        detection=detection,
        rewrite_result=rewrite_result,
        extended_signals=batch_result.extended_signals
        or (fallback_plan.extended_signals if fallback_plan else None),
        keywords=batch_result.keywords or (fallback_plan.keywords if fallback_plan else []),
    )


def _merge_rewrite_results(
    preferred: RewriteResult | None,
    deterministic: RewriteResult | None,
) -> RewriteResult | None:
    """Keep explicit deterministic decomposition unless a model supplies one."""
    if (
        deterministic is None
        or not deterministic.is_compound
        or not deterministic.decomposed_queries
    ):
        return preferred
    if preferred is None:
        return deterministic
    if preferred.is_compound and preferred.decomposed_queries:
        return preferred

    rewrite_type = (
        RewriteType.DECOMPOSITION
        if preferred.rewrite_type in {RewriteType.NONE, RewriteType.DECOMPOSITION}
        else RewriteType.COMBINED
    )
    metadata = dict(preferred.metadata)
    metadata["decomposition_source"] = deterministic.metadata.get(
        "source",
        "deterministic",
    )
    return replace(
        preferred,
        rewrite_type=rewrite_type,
        confidence=max(preferred.confidence, deterministic.confidence),
        is_compound=True,
        decomposed_queries=list(deterministic.decomposed_queries),
        metadata=metadata,
    )


def deterministic_analysis(query: str, terms: list[str] | None = None) -> QueryAnalysis:
    """Classify the query's likely retrieval target from explicit terms."""
    lowered_terms = {t.lower() for t in (terms or content_terms(query))}
    lower = query.lower()

    if lowered_terms & _DATA_TERMS:
        primary = QueryType.DATA
    elif lowered_terms & _CODE_TERMS:
        primary = QueryType.CODE
    elif lowered_terms & _DOC_TERMS:
        primary = QueryType.DOCUMENTATION
    elif (" code " in f" {lower} " or "::" in query or "." in query) and any(
        t in lower for t in ("where", "defined", "implementation")
    ):
        primary = QueryType.CODE
    else:
        primary = QueryType.GENERAL

    return QueryAnalysis(
        primary_type=primary,
        confidence=0.65,
        entities=tuple((terms or content_terms(query))[:8]),
        refined_query=query,
    )


def deterministic_detection(query: str) -> DetectionSummary:
    """Detect common retrieval intents with fixed rules and module parsers."""
    return DetectionSummary(
        temporal=_temporal_detection(query),
        aggregation=_aggregation_detection(query),
        comparison=_comparison_detection(query),
        freshness=_freshness_detection(query),
    )


def deterministic_keywords(query: str, terms: list[str] | None = None) -> list[str]:
    """Extract literal query terms for deterministic retrieval."""
    seen: set[str] = set()
    keywords: list[str] = []
    for term in terms or content_terms(query):
        low = term.lower()
        if low not in seen:
            seen.add(low)
            keywords.append(term)
    return keywords


def content_terms(query: str) -> list[str]:
    """Return meaningful lexical terms from a query."""
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        normalized = token.lower()
        if len(normalized) <= 2 or normalized in _STOP_WORDS:
            continue
        terms.append(token)
    return terms


def _deterministic_decomposition(query: str) -> RewriteResult | None:
    """Split only explicit multi-question syntax into focused retrieval legs."""
    raw_parts = _EXPLICIT_CLAUSE_BOUNDARY.split(query.strip())
    parts: list[str] = []
    seen: set[str] = set()
    for raw_part in raw_parts:
        part = re.sub(r"^\s*[-*]\s*", "", raw_part).strip(" \t\r\n?;,")
        key = part.casefold()
        if len(part) < 3 or not content_terms(part) or key in seen:
            continue
        seen.add(key)
        parts.append(part)

    if len(parts) < 2:
        return None
    return RewriteResult(
        original_query=query,
        rewritten_query=query,
        rewrite_type=RewriteType.DECOMPOSITION,
        confidence=1.0,
        is_compound=True,
        decomposed_queries=parts[:5],
        metadata={"source": "deterministic_explicit_clauses"},
    )


def _temporal_detection(query: str):
    lower = query.lower()
    references = _temporal_references(lower)
    if not references:
        return TemporalModule().not_detected()

    if re.search(r"\b(between|since|before|after|from|to)\b", lower):
        intent = TemporalIntent.RANGE.value
    elif any(ref in lower for ref in ("trend", "over time", "history")):
        intent = TemporalIntent.TREND.value
    else:
        intent = TemporalIntent.POINT_IN_TIME.value

    return TemporalModule().parse_result(
        {
            "detected": True,
            "intent": intent,
            "references": references,
            "time_focused_queries": _time_focused_queries(query, references),
        }
    )


def _aggregation_detection(query: str):
    lower = query.lower()
    scalar_measurement = re.search(
        r"\bhow many\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
        r"bytes?|kilobytes?|megabytes?|gigabytes?)\s+(?:is|does)\s+(?:the|a|an)\b",
        lower,
    )
    if scalar_measurement:
        return AggregationModule().not_detected()
    if re.search(r"\b(how many|count|number of)\b", lower):
        return AggregationModule().parse_result(
            {
                "detected": True,
                "type": AggregationType.COUNT.value,
                "target": None,
                "fetch_multiplier": 4,
            }
        )
    if re.search(
        r"^\s*(?:list|enumerate)\b|"
        r"\b(list|show|find|identify|return|provide|give me)?\s*"
        r"(?:all|every|each|complete inventory|full set|entire set)\b|"
        r"\b(?:enumerate|unique|distinct)\b",
        lower,
    ):
        agg_type = (
            AggregationType.UNIQUE
            if re.search(r"\b(unique|distinct)\b", lower)
            else AggregationType.LIST
        )
        return AggregationModule().parse_result(
            {
                "detected": True,
                "type": agg_type.value,
                "target": None,
                "fetch_multiplier": 3,
            }
        )
    if re.search(
        r"^\s*(?:what|which)\s+(?:[a-z0-9_-]+\s+){0,3}"
        r"[a-z][a-z0-9_-]*s\s+"
        r"(?:are|were|have|had|do|did|"
        r"affected|apply|exist|include|contain|use|support|belong)\b",
        lower,
    ):
        return AggregationModule().parse_result(
            {
                "detected": True,
                "type": AggregationType.LIST.value,
                "target": None,
                "fetch_multiplier": 3,
            }
        )
    return AggregationModule().not_detected()


def _comparison_detection(query: str):
    lower = query.lower()
    entities = _comparison_entities(query)
    how_different = re.search(
        r"^\s*how\s+(?:is|are|was|were)\b.{0,120}\bdifferent\s*[?.]?\s*$",
        lower,
    )
    detected = bool(entities) or bool(re.search(_COMPARATIVE_PATTERN, lower)) or bool(how_different)
    if not detected:
        return ComparisonModule().not_detected()

    return ComparisonModule().parse_result(
        {
            "detected": True,
            "entities": entities,
            "comparison_queries": _comparison_queries(query, entities),
        }
    )


def _freshness_detection(query: str):
    detected = bool(
        re.search(r"\b(latest|recent|newest|current|today|fresh|updated)\b", query.lower())
    )
    if not detected:
        return FreshnessModule().not_detected()
    return FreshnessModule().parse_result({"detected": True})


def _temporal_references(lower_query: str) -> list[str]:
    patterns = (
        r"\bq[1-4](?:\s+\d{4})?\b",
        _MONTH_PATTERN,
        r"(?<![\d.])\b\d{4}\b(?![\d.])",
        r"\b(?:last|next)\s+(?:week|month|quarter|year)\b",
        r"\b(?:latest|newest|current|final|recent|recently|most recent|most recently)\b",
        r"\b(?:today|yesterday|tomorrow)\b",
        r"\b(?:since|before|after)\s+[A-Za-z0-9_-]+\b",
        r"\b(?:as of|at that time|effective now)\b",
        r"\b(?:currently|now|previous|prior|original|superseded|formerly)\b",
        r"\b(?:at|during)\s+(?:the\s+)?"
        r"(?:launch|release|deployment|migration|rollout|incident|outage|cutover)\b",
        r"\bwhen\s+(?:[A-Za-z0-9_.-]+\s+){0,4}"
        r"(?:launched|released|deployed|migrated|started|ended|failed|recovered)\b",
    )
    seen: set[str] = set()
    refs: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, lower_query):
            ref = match.group(0).strip()
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def _time_focused_queries(query: str, references: list[str]) -> list[str]:
    if len(references) <= 1:
        return []

    non_temporal_terms = [
        term
        for term in content_terms(query)
        if term.lower() not in {part for ref in references for part in ref.split()}
    ]
    tail = " ".join(non_temporal_terms[:6])
    return [f"{ref} {tail}".strip() for ref in references if ref]


def _comparison_entities(query: str) -> list[str]:
    lower = query.lower()
    patterns = (
        r"\b(?:compare|comparing)\s+(.+?)\s+(?:and|with|to|vs|versus)\s+(.+?)(?:[?.!,]|$)",
        r"\bdifference between\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)",
        r"\b(?:changed|changes|change)\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)",
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[?.!,]|$)",
        r"\b(.+?)\s+(?:vs|versus)\s+(.+?)(?:[?.!,]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        entities = [_clean_entity(match.group(1)), _clean_entity(match.group(2))]
        return [entity for entity in entities if entity]

    if re.search(_COMPARATIVE_PATTERN, lower):
        temporal_refs = _temporal_references(lower)
        if len(temporal_refs) >= 2:
            return temporal_refs[:4]

        or_match = re.search(
            r"\b([A-Za-z0-9][A-Za-z0-9_.-]*)\s+or\s+([A-Za-z0-9][A-Za-z0-9_.-]*)\b",
            query,
            flags=re.IGNORECASE,
        )
        if or_match:
            entities = [_clean_entity(or_match.group(1)), _clean_entity(or_match.group(2))]
            return [entity for entity in entities if entity]
    return []


def _comparison_queries(query: str, entities: list[str]) -> list[str]:
    if not entities:
        return []

    query_terms = " ".join(content_terms(query)[:6])
    queries = [f"{entity} {query_terms}".strip() for entity in entities]
    queries.append(" ".join([*entities, query_terms]).strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for item in queries:
        low = item.lower()
        if item and low not in seen:
            seen.add(low)
            deduped.append(item)
    return deduped


def _clean_entity(value: str) -> str:
    cleaned = re.sub(r"^\s*(compare|difference|between|the)\s+", "", value, flags=re.IGNORECASE)
    return cleaned.strip(" ?.,")


def _specificity(query: str, detection: DetectionSummary | None) -> str:
    lower = query.lower()
    if detection and (
        detection.has_aggregation_intent
        or detection.has_comparison_intent
        or detection.has_temporal_intent
    ):
        return "moderate"
    if re.search(
        r"\b(overview|summarize|summary|main themes|key facts|all|everything|survey)\b",
        lower,
    ):
        return "broad"
    if len(content_terms(query)) <= 4:
        return "narrow"
    return "moderate"


def _answer_type(query: str, detection: DetectionSummary | None) -> str:
    lower = query.lower()
    if detection and detection.has_comparison_intent:
        return "comparative"
    if re.search(r"\b(how do|how to|steps|procedure|workflow)\b", lower):
        return "procedural"
    if re.search(r"\b(overview|summarize|summary|key facts|explore|survey)\b", lower):
        return "exploratory"
    return "factual"


def _domain(query: str, terms: list[str]) -> str:
    lower = query.lower()
    lowered_terms = {t.lower() for t in terms}
    if lowered_terms & (_CODE_TERMS | _DATA_TERMS) or re.search(r"\b(api|sql|json|http)\b", lower):
        return "technical"
    if re.search(r"\b(contract|policy|compliance|legal|regulation)\b", lower):
        return "legal"
    if re.search(r"\b(revenue|cost|budget|invoice|financial)\b", lower):
        return "financial"
    if re.search(r"\b(patient|clinical|medical|diagnosis)\b", lower):
        return "medical"
    return "general"


__all__ = [
    "DeterministicQueryPlanner",
    "QueryPlan",
    "content_terms",
    "deterministic_analysis",
    "deterministic_detection",
    "deterministic_keywords",
    "plan_from_batch_result",
]
