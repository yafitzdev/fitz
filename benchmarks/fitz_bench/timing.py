"""Shared aggregation for exclusive Fitz-Sage query-stage timings."""

from __future__ import annotations

import re
from typing import Any

from benchmarks.fitz_bench.retrieval_eval import summarize_latency

_AGGREGATE_TIMING = re.compile(r"^Evidence closure \d+$")
_STANDARD_GROUPS = (
    "query_prep",
    "semantic_expansion",
    "pyrrho_planning",
    "recall",
    "rerank",
    "read",
    "context_expansion",
    "table_queries",
    "evidence_compiler",
    "pyrrho_decision",
    "other",
    "unattributed",
)


def group_timings(
    timings: dict[str, float],
    *,
    total_seconds: float,
) -> tuple[dict[str, float], float]:
    """Collapse non-overlapping engine timings into comparable stage groups."""
    grouped: dict[str, float] = {}
    for name, duration in timings.items():
        group = timing_group(name)
        if group is None:
            continue
        grouped[group] = grouped.get(group, 0.0) + float(duration)

    accounted = sum(grouped.values())
    grouped["unattributed"] = max(0.0, total_seconds - accounted)
    overlap = max(0.0, accounted - total_seconds)
    return grouped, overlap


def timing_group(name: str) -> str | None:
    """Map one raw timing label to an exclusive group, ignoring aggregate totals."""
    if name == "Retrieval" or _AGGREGATE_TIMING.fullmatch(name):
        return None
    if name == "Query prep":
        return "query_prep"
    if name == "Qwen query keywords":
        return "semantic_expansion"
    if name == "Pyrrho pre":
        return "pyrrho_planning"
    if name == "Pyrrho":
        return "pyrrho_decision"
    if name == "Recall" or name.endswith(" Recall"):
        return "recall"
    if name == "Rerank" or name.endswith(" Rerank"):
        return "rerank"
    if name == "Read" or name.endswith(" Read"):
        return "read"
    if name == "Expand context" or name.endswith(" expand"):
        return "context_expansion"
    if name == "Table queries" or name.endswith(" table queries"):
        return "table_queries"
    if "compile" in name.casefold():
        return "evidence_compiler"
    return "other"


def summarize_timing_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize records containing total, grouped, and raw stage durations."""
    if not records:
        return {}
    totals = [float(record["total_seconds"]) for record in records]
    total_sum = sum(totals)
    group_names = list(_STANDARD_GROUPS)
    extras = sorted(
        {
            group
            for record in records
            for group in record["grouped_seconds"]
            if group not in group_names
        }
    )
    group_names.extend(extras)

    stage_groups: dict[str, dict[str, float]] = {}
    for group in group_names:
        values = [float(record["grouped_seconds"].get(group, 0.0)) for record in records]
        if not any(values):
            continue
        group_summary = summarize_latency(values)
        group_summary["share_of_total"] = sum(values) / total_sum if total_sum else 0.0
        stage_groups[group] = group_summary

    raw_names = sorted({name for record in records for name in record["stage_seconds"]})
    raw_stages: dict[str, dict[str, float | int]] = {}
    for name in raw_names:
        values = [
            float(record["stage_seconds"][name])
            for record in records
            if name in record["stage_seconds"]
        ]
        raw_summary: dict[str, float | int] = summarize_latency(values)
        raw_summary["observations"] = len(values)
        raw_summary["occurrence_rate"] = len(values) / len(records)
        raw_stages[name] = raw_summary

    return {
        "runs": len(records),
        "total_latency": summarize_latency(totals),
        "stage_groups": stage_groups,
        "raw_stages": raw_stages,
    }
