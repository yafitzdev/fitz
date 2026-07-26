"""Stage-level failure attribution for evidence-first benchmark cases."""

from __future__ import annotations

from typing import Any

from benchmarks.fitz_bench.models import BenchmarkCase, EvidenceExpectation, ValidationResult
from benchmarks.fitz_bench.validators import _matches
from fitz_sage.core import RetrievalRun


def diagnose_case(
    case: BenchmarkCase,
    validation: ValidationResult,
    run: RetrievalRun,
) -> dict[str, Any]:
    """Attribute benchmark failures to the narrowest observable pipeline stage."""
    issues: list[dict[str, Any]] = []
    for index, rank in enumerate(validation.matched_required):
        if rank is not None:
            continue
        expectation = case.required_evidence[index]
        issues.append(
            {
                "type": "missing_required_evidence",
                "expectation_index": index,
                "stage": _missing_stage(expectation, run),
                "expectation": _expectation_dict(expectation),
            }
        )

    for rank in validation.matched_forbidden:
        issues.append(
            {
                "type": "forbidden_evidence",
                "stage": "pre_governance_selection",
                "rank": rank,
            }
        )
    for index, rank in enumerate(validation.matched_delivery_required):
        if rank is not None or validation.matched_required[index] is None:
            continue
        issues.append(
            {
                "type": "missing_delivered_evidence",
                "expectation_index": index,
                "stage": "governance_cutoff",
                "expectation": _expectation_dict(case.required_evidence[index]),
            }
        )
    for rank in validation.matched_delivery_forbidden:
        issues.append(
            {
                "type": "forbidden_delivered_evidence",
                "stage": "governed_delivery",
                "rank": rank,
            }
        )

    if validation.governance_failures:
        issues.append(
            {
                "type": "governance_mode",
                "stage": "governance",
                "expected": case.expected_mode,
                "actual": run.governance.mode,
            }
        )
    for failure in validation.query_shape_failures:
        issues.append(
            {
                "type": "query_shape_signal",
                "stage": "query_interpretation",
                "detail": failure,
            }
        )

    counts: dict[str, int] = {}
    for issue in issues:
        stage = str(issue["stage"])
        counts[stage] = counts.get(stage, 0) + 1
    return {"issues": issues, "by_stage": counts}


def compact_run(run: RetrievalRun) -> dict[str, Any]:
    """Return a compact debug record without source content or bulky raw traces."""
    return {
        "run_id": run.run_id,
        "mode": _mode_value(run),
        "reasons": list(run.evidence.reasons),
        "query": run.query.to_dict(),
        "evidence": [
            {
                "rank": item.rank,
                "file_path": item.file_path,
                "source_id": item.source_id,
                "address_kind": item.address_kind,
                "address_location": item.address_location,
                "score": item.score,
            }
            for item in run.evidence.items
        ],
        "ranked_evidence": [
            {
                "rank": item.rank,
                "file_path": item.file_path,
                "source_id": item.source_id,
                "address_kind": item.address_kind,
                "address_location": item.address_location,
                "score": item.score,
                "content_sha256": item.content_sha256,
            }
            for item in run.ranked_evidence
        ],
        "candidate_stages": [
            {"name": stage.name, "count": len(stage.candidates)} for stage in run.candidate_stages
        ],
        "governance": run.governance.to_dict(),
        "environment": run.environment.to_dict(),
        "warnings": list(run.warnings),
    }


def evidence_signature(run: RetrievalRun) -> dict[str, Any]:
    """Return the stable observable result used by reload checks."""
    return {
        "mode": _mode_value(run),
        "retrieval": [
            (
                item.file_path.replace("\\", "/").lower(),
                item.address_kind,
                item.address_location,
            )
            for item in run.ranked_evidence
        ],
        "delivery": [
            (
                item.file_path.replace("\\", "/").lower(),
                item.address_kind,
                item.address_location,
            )
            for item in run.evidence.items
        ],
    }


def _missing_stage(expectation: EvidenceExpectation, run: RetrievalRun) -> str:
    final_items = [item.to_dict() for item in run.evidence.items]
    if any(_scope_matches(expectation, item) for item in final_items):
        return "evidence_content"

    ranked_items = [
        {
            "file_path": item.file_path,
            "source_id": item.source_id,
            "address_kind": item.address_kind,
            "address_location": item.address_location,
            "content": item.content or "",
            "excerpt": "",
        }
        for item in run.ranked_evidence
    ]
    if any(_matches(expectation, item) for item in ranked_items):
        return "governance_cutoff"
    if any(_scope_matches(expectation, item) for item in ranked_items):
        return "evidence_content"

    trace = run.evidence.metadata.get("retrieval_trace", {})
    if not isinstance(trace, dict):
        return "recall"
    reranker = trace.get("reranker")
    reranked = reranker.get("output") if isinstance(reranker, dict) else None
    stages = (
        ("evidence_read_or_compilation", trace.get("final_addresses")),
        ("final_selection", reranked),
        ("reranking", trace.get("recall")),
    )
    for label, candidates in stages:
        if isinstance(candidates, list) and any(
            _candidate_scope_matches(expectation, candidate)
            for candidate in candidates
            if isinstance(candidate, dict)
        ):
            return label
    return "recall"


def _scope_matches(expectation: EvidenceExpectation, item: dict[str, Any]) -> bool:
    scope_only = EvidenceExpectation(
        file=expectation.file,
        kind=expectation.kind,
        location_contains=expectation.location_contains,
    )
    return _matches(scope_only, item)


def _candidate_scope_matches(
    expectation: EvidenceExpectation,
    candidate: dict[str, Any],
) -> bool:
    if expectation.kind and str(candidate.get("kind") or "") != expectation.kind:
        return False

    location = str(candidate.get("location") or "").replace("\\", "/").lower()
    if expectation.location_contains:
        if expectation.location_contains.lower() not in location:
            return False

    if not expectation.file:
        return True

    expected = expectation.file.replace("\\", "/").lower()
    metadata = candidate.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    paths = [
        str(candidate.get("file_path") or ""),
        str(metadata.get("source_path") or ""),
        str(metadata.get("disk_path") or ""),
    ]
    if any(expected in path.replace("\\", "/").lower() for path in paths):
        return True

    expected_module = expected.rsplit(".", 1)[0].replace("/", ".")
    return bool(expected_module and expected_module in location.replace("/", "."))


def _expectation_dict(expectation: EvidenceExpectation) -> dict[str, Any]:
    return {
        "file": expectation.file,
        "kind": expectation.kind,
        "location_contains": expectation.location_contains,
        "contains": list(expectation.contains),
        "contains_any": list(expectation.contains_any),
    }


def _mode_value(run: RetrievalRun) -> str:
    mode = run.evidence.mode
    return str(getattr(mode, "value", mode) or "unknown")


__all__ = ["compact_run", "diagnose_case", "evidence_signature"]
