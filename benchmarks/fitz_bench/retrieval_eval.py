"""Dependency-free document retrieval baseline and graded ranking metrics."""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Apply the baseline's explicit, language-neutral lexical analyzer."""
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(text)]


@dataclass
class PlainBm25:
    """Small transparent Okapi BM25 implementation for benchmark comparison."""

    document_ids: list[str]
    document_lengths: list[int]
    postings: dict[str, list[tuple[int, int]]]
    average_document_length: float
    k1: float = 1.2
    b: float = 0.75

    @classmethod
    def build(
        cls,
        documents: Iterable[tuple[str, str]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> "PlainBm25":
        document_ids: list[str] = []
        document_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        seen: set[str] = set()
        for index, (document_id, text) in enumerate(documents):
            if document_id in seen:
                raise ValueError(f"Duplicate BM25 document ID: {document_id}")
            seen.add(document_id)
            terms = tokenize(text)
            document_ids.append(document_id)
            document_lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                postings.setdefault(term, []).append((index, frequency))
        average = sum(document_lengths) / len(document_lengths) if document_lengths else 0.0
        return cls(
            document_ids=document_ids,
            document_lengths=document_lengths,
            postings=postings,
            average_document_length=average,
            k1=k1,
            b=b,
        )

    def search(self, query: str, *, top_k: int) -> list[str]:
        """Return document IDs ordered by descending Okapi BM25 score."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        document_count = len(self.document_ids)
        if not document_count or self.average_document_length <= 0:
            return []
        scores: dict[int, float] = {}
        for term, query_frequency in Counter(tokenize(query)).items():
            term_postings = self.postings.get(term)
            if not term_postings:
                continue
            document_frequency = len(term_postings)
            inverse_document_frequency = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_index, term_frequency in term_postings:
                normalized_length = (
                    self.document_lengths[document_index] / self.average_document_length
                )
                denominator = term_frequency + self.k1 * (1.0 - self.b + self.b * normalized_length)
                term_score = (
                    inverse_document_frequency * term_frequency * (self.k1 + 1.0) / denominator
                )
                scores[document_index] = (
                    scores.get(document_index, 0.0) + query_frequency * term_score
                )
        best = heapq.nsmallest(
            top_k,
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [self.document_ids[index] for index, _score in best]


def ranking_metrics(
    ranked_document_ids: Sequence[str],
    judgments: Mapping[str, int],
    cutoffs: Sequence[int],
) -> dict[str, float]:
    """Compute TREC-style graded metrics at each requested cutoff."""
    normalized_cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    if not normalized_cutoffs or normalized_cutoffs[0] < 1:
        raise ValueError("Metric cutoffs must be positive.")
    relevant = {document_id for document_id, score in judgments.items() if score > 0}
    ranked = _deduplicate(ranked_document_ids)
    output: dict[str, float] = {}
    for cutoff in normalized_cutoffs:
        selected = ranked[:cutoff]
        hit_count = sum(document_id in relevant for document_id in selected)
        output[f"Precision@{cutoff}"] = hit_count / cutoff
        output[f"Recall@{cutoff}"] = hit_count / len(relevant) if relevant else 0.0
        output[f"MRR@{cutoff}"] = _reciprocal_rank(selected, relevant)
        output[f"MAP@{cutoff}"] = _average_precision(selected, relevant, cutoff=cutoff)
        output[f"NDCG@{cutoff}"] = _ndcg(selected, judgments, cutoff=cutoff)
    return output


def aggregate_metrics(records: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Mean metric values across query records."""
    totals: dict[str, float] = {}
    count = 0
    for record in records:
        count += 1
        for key, value in record.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    if count == 0:
        return {}
    return {key: value / count for key, value in sorted(totals.items())}


def stage_failure(
    stages: Mapping[str, Sequence[str]],
    judgments: Mapping[str, int],
) -> str:
    """Attribute a delivered miss to the earliest irreversible package boundary."""
    relevant = {document_id for document_id, score in judgments.items() if score > 0}
    for name in ("recall", "final", "compiled", "delivered"):
        if not relevant.intersection(stages.get(name, ())):
            return name
    return "delivered_hit"


def stage_recoveries(
    stages: Mapping[str, Sequence[str]],
    judgments: Mapping[str, int],
) -> list[str]:
    """Describe relevant-document recoveries between non-monotonic stages."""
    relevant = {document_id for document_id, score in judgments.items() if score > 0}
    recoveries: list[str] = []
    if not relevant.intersection(stages.get("reranked", ())) and relevant.intersection(
        stages.get("final", ())
    ):
        recoveries.append("final_rescued_reranker_miss")
    return recoveries


def metric_delta(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
) -> dict[str, float]:
    """Subtract baseline metrics from same-named candidate metrics."""
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in sorted(set(candidate).intersection(baseline))
    }


def summarize_latency(durations: Sequence[float]) -> dict[str, float]:
    """Return compact latency statistics without external dependencies."""
    if not durations:
        return {}
    ordered = sorted(float(value) for value in durations)
    return {
        "mean_seconds": sum(ordered) / len(ordered),
        "p50_seconds": _percentile(ordered, 0.50),
        "p95_seconds": _percentile(ordered, 0.95),
        "max_seconds": ordered[-1],
    }


def _deduplicate(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for rank, document_id in enumerate(ranked, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def _average_precision(
    ranked: Sequence[str],
    relevant: set[str],
    *,
    cutoff: int,
) -> float:
    if not relevant:
        return 0.0
    precision_sum = 0.0
    hits = 0
    for rank, document_id in enumerate(ranked, start=1):
        if document_id not in relevant:
            continue
        hits += 1
        precision_sum += hits / rank
    return precision_sum / min(len(relevant), cutoff)


def _ndcg(
    ranked: Sequence[str],
    judgments: Mapping[str, int],
    *,
    cutoff: int,
) -> float:
    actual = sum(
        max(0, int(judgments.get(document_id, 0))) / math.log2(rank + 1)
        for rank, document_id in enumerate(ranked, start=1)
    )
    ideal_scores = sorted(
        (max(0, int(value)) for value in judgments.values()),
        reverse=True,
    )[:cutoff]
    ideal = sum(score / math.log2(rank + 1) for rank, score in enumerate(ideal_scores, start=1))
    return actual / ideal if ideal else 0.0


def _percentile(ordered: Sequence[float], fraction: float) -> float:
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def metric_formulas() -> dict[str, Any]:
    """Machine-readable semantics included in every benchmark report."""
    return {
        "relevance": "qrel score > 0",
        "deduplication": "first occurrence of each document ID",
        "ndcg_gain": "linear qrel score",
        "map_denominator": "min(number of relevant documents, cutoff)",
        "precision_denominator": "cutoff",
        "aggregation": "unweighted arithmetic mean across judged test queries",
    }
