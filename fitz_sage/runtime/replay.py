"""Governance replay over content-bearing retrieval-run records."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fitz_sage.core import (
    EvidenceItem,
    EvidencePack,
    GovernanceReplay,
    RetrievalRun,
)
from fitz_sage.engines.fitz_krag.governance_cutoff import apply_governance_cutoff
from fitz_sage.engines.fitz_krag.run_trace import governance_execution
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from fitz_sage.governance import create_governance


def load_retrieval_run(value: RetrievalRun | str | Path) -> RetrievalRun:
    """Return a retrieval run from an object, path, or JSON string."""
    if isinstance(value, RetrievalRun):
        return value
    if isinstance(value, Path):
        return RetrievalRun.read(value)
    candidate = Path(value).expanduser()
    try:
        if candidate.is_file():
            return RetrievalRun.read(candidate)
    except (OSError, ValueError):
        pass
    return RetrievalRun.from_json(value)


def replay_governance(
    value: RetrievalRun | str | Path,
    governance: str | Any | None = None,
) -> GovernanceReplay:
    """Re-evaluate a frozen ranked evidence set with a governance provider.

    This intentionally does not rerun query preparation, retrieval, reranking,
    evidence closure, or compilation. The trace must have been exported with
    content and every content digest must still match.
    """
    run = load_retrieval_run(value)
    _validate_replayable(run)
    classifier, spec = _resolve_governance(run, governance)
    results = [_read_result(item) for item in run.ranked_evidence]
    profile = _replay_profile(run)

    started = time.perf_counter()
    cutoff = apply_governance_cutoff(
        run.query.sanitized_text,
        results,
        classifier,
        profile=profile,
        requested_top_k=run.governance.max_documents or None,
    )
    duration = time.perf_counter() - started
    replayed = governance_execution(cutoff.metadata, cutoff.reasons)
    pack = EvidencePack(
        query=run.query.sanitized_text,
        mode=cutoff.mode,
        items=[
            _evidence_item(result, rank) for rank, result in enumerate(cutoff.selected, start=1)
        ],
        reasons=list(cutoff.reasons),
        timings={"Governance replay": duration},
        indexing_status=dict(run.environment.indexing_status),
        metadata={
            "engine": run.environment.engine,
            "source_run_id": run.run_id,
            "governance_spec": spec,
            "governance_cutoff": cutoff.metadata,
        },
    )
    return GovernanceReplay(
        replay_id=str(uuid.uuid4()),
        source_run_id=run.run_id,
        created_at=_utc_now(),
        governance_spec=spec,
        source_fitz_sage_version=run.environment.fitz_sage_version,
        replay_fitz_sage_version=_fitz_version(),
        original=run.governance,
        replayed=replayed,
        evidence=pack,
        duration_seconds=duration,
    )


def _validate_replayable(run: RetrievalRun) -> None:
    if run.environment.engine != "fitz_krag":
        raise ValueError("Governance replay currently supports fitz_krag retrieval runs only.")
    if not run.content_included:
        raise ValueError("Governance replay requires a trace exported with source content.")
    if not run.ranked_evidence:
        raise ValueError("Governance replay requires frozen ranked evidence.")
    invalid = [item.rank for item in run.ranked_evidence if not item.verify_content()]
    if invalid:
        raise ValueError(
            "Frozen evidence failed integrity checks at ranks "
            + ", ".join(str(rank) for rank in invalid)
            + "."
        )


def _resolve_governance(
    run: RetrievalRun,
    governance: str | Any | None,
) -> tuple[Any, str]:
    if governance is None:
        spec = run.environment.components.get("governance")
        if not spec:
            raise ValueError(
                "The trace does not identify a governance provider; "
                "pass governance='pyrrho/<package>'."
            )
        return create_governance(spec), spec
    if isinstance(governance, str):
        return create_governance(governance), governance
    if not callable(getattr(governance, "decide", None)):
        raise TypeError("governance must be a provider spec or expose decide().")
    spec = str(
        getattr(governance, "_model_id", None)
        or f"{type(governance).__module__}.{type(governance).__qualname__}"
    )
    return governance, spec


def _read_result(item: Any) -> ReadResult:
    try:
        kind = AddressKind(item.address_kind)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported frozen evidence kind at rank {item.rank}: " f"{item.address_kind!r}."
        ) from exc
    metadata = {}
    if item.compiler_metadata:
        metadata["evidence_compiler"] = dict(item.compiler_metadata)
    address = Address(
        kind=kind,
        source_id=item.source_id,
        location=item.address_location,
        summary=item.address_summary,
        score=item.score or 0.0,
        metadata={},
    )
    return ReadResult(
        address=address,
        content=item.content or "",
        file_path=item.file_path,
        line_range=item.line_range,
        metadata=metadata,
    )


def _evidence_item(result: ReadResult, rank: int) -> EvidenceItem:
    kind = getattr(result.address.kind, "value", str(result.address.kind))
    text = " ".join(result.content.split())
    excerpt = text if len(text) <= 320 else text[:317].rstrip() + "..."
    return EvidenceItem(
        rank=rank,
        source_id=result.address.source_id,
        file_path=result.file_path,
        address_kind=kind,
        address_location=result.address.location,
        line_range=result.line_range,
        score=result.address.score,
        excerpt=excerpt,
        content=result.content,
        metadata={**result.address.metadata, **result.metadata},
    )


def _replay_profile(run: RetrievalRun) -> SimpleNamespace:
    query = run.query
    return SimpleNamespace(
        query_contract=query.query_contract,
        specificity=query.specificity,
        answer_type=query.answer_type,
        retrieval_modality=query.retrieval_modality,
        required_modalities=query.required_modalities,
        comparison_queries=query.comparison_queries,
        comparison_entities=query.comparison_entities,
        has_comparison_intent=query.has_comparison_intent,
        has_aggregation_intent=query.has_aggregation_intent,
        inject_corpus_summaries=query.inject_corpus_summaries,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fitz_version() -> str:
    try:
        from importlib.metadata import version

        return version("fitz-sage")
    except Exception:
        from fitz_sage import __version__

        return __version__


__all__ = ["load_retrieval_run", "replay_governance"]
