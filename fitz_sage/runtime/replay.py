"""Pyrrho replay over content-bearing retrieval-run records."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitz_sage.core import (
    EvidenceItem,
    EvidencePack,
    PyrrhoReplay,
    RetrievalRun,
)
from fitz_sage.engines.fitz_krag.run_trace import pyrrho_execution
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from fitz_sage.integrations.pyrrho import (
    answer_mode_from_pyrrho,
    create_pyrrho,
    decide,
    decision_payload,
)


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


def replay_pyrrho(
    value: RetrievalRun | str | Path,
    pyrrho: str | Any | None = None,
) -> PyrrhoReplay:
    """Re-evaluate the exact frozen evidence set with Pyrrho.

    This intentionally does not rerun query preparation, retrieval, reranking,
    evidence closure, or compilation. The trace must have been exported with
    content and every content digest must still match.
    """
    run = load_retrieval_run(value)
    _validate_replayable(run)
    runtime, spec = _resolve_pyrrho(run, pyrrho)
    results = [_read_result(item) for item in run.pyrrho_evidence]

    started = time.perf_counter()
    decision = decide(runtime, run.query.sanitized_text, results)
    duration = time.perf_counter() - started
    replayed = pyrrho_execution(decision, evidence_count=len(results))
    mode = answer_mode_from_pyrrho(decision)
    pack = EvidencePack(
        query=run.query.sanitized_text,
        mode=mode,
        items=[_evidence_item(result, rank) for rank, result in enumerate(results, start=1)],
        reasons=list(decision.reasons),
        timings={"Pyrrho replay": duration},
        indexing_status=dict(run.environment.indexing_status),
        metadata={
            "engine": run.environment.engine,
            "source_run_id": run.run_id,
            "pyrrho_spec": spec,
            "pyrrho": decision_payload(decision),
        },
    )
    return PyrrhoReplay(
        replay_id=str(uuid.uuid4()),
        source_run_id=run.run_id,
        created_at=_utc_now(),
        pyrrho_spec=spec,
        source_fitz_sage_version=run.environment.fitz_sage_version,
        replay_fitz_sage_version=_fitz_version(),
        original=run.pyrrho,
        replayed=replayed,
        evidence=pack,
        duration_seconds=duration,
    )


def _validate_replayable(run: RetrievalRun) -> None:
    if run.environment.engine != "fitz_krag":
        raise ValueError("Pyrrho replay currently supports fitz_krag retrieval runs only.")
    if not run.content_included:
        raise ValueError("Pyrrho replay requires a trace exported with source content.")
    invalid = [item.rank for item in run.pyrrho_evidence if not item.verify_content()]
    if invalid:
        raise ValueError(
            "Frozen evidence failed integrity checks at ranks "
            + ", ".join(str(rank) for rank in invalid)
            + "."
        )


def _resolve_pyrrho(
    run: RetrievalRun,
    pyrrho: str | Any | None,
) -> tuple[Any, str]:
    if pyrrho is None:
        spec = run.environment.components.get("pyrrho")
        if not spec:
            raise ValueError("The trace does not identify Pyrrho; " "pass pyrrho='pyrrho/<model>'.")
        return create_pyrrho(spec), spec
    if isinstance(pyrrho, str):
        return create_pyrrho(pyrrho), pyrrho
    if not callable(getattr(pyrrho, "decide", None)):
        raise TypeError("pyrrho must be a provider spec or expose decide().")
    spec = str(
        getattr(pyrrho, "model_spec", None)
        or f"{type(pyrrho).__module__}.{type(pyrrho).__qualname__}"
    )
    return pyrrho, spec


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fitz_version() -> str:
    try:
        from importlib.metadata import version

        return version("fitz-sage")
    except Exception:
        from fitz_sage import __version__

        return __version__


__all__ = ["load_retrieval_run", "replay_pyrrho"]
