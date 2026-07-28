"""Convert KRAG internals into the stable public retrieval-run schema."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitz_sage.core import (
    CandidateReference,
    CandidateStage,
    FrozenEvidence,
    PyrrhoExecution,
    QueryExecution,
    QueryTerm,
    RetrievalRun,
    RunEnvironment,
    StrategyExecution,
)

_SENSITIVE_CONFIG_PARTS = ("api_key", "auth", "certificate", "cert_path", "secret", "token")


def build_retrieval_run(
    *,
    source_query: str,
    pack: Any,
    outcome: Any,
    compilation: Any,
    selected: Any,
    decision: Any,
    config: Any,
    indexing_status: dict[str, Any],
    workspace: Path,
) -> RetrievalRun:
    """Build one public execution record from the canonical governed result."""
    profile = getattr(outcome, "profile", None)
    query_shape = _query_shape(profile)
    query = QueryExecution(
        source_text=source_query,
        sanitized_text=str(getattr(outcome, "sanitized", "") or ""),
        retrieval_text=str(getattr(outcome, "retrieval_query", "") or ""),
        query_shape=query_shape,
        query_contract=_optional_profile_text(profile, "query_contract"),
        specificity=_optional_profile_text(profile, "specificity"),
        answer_type=_optional_profile_text(profile, "answer_type"),
        retrieval_modality=_optional_profile_text(profile, "retrieval_modality"),
        required_modalities=_profile_tuple(profile, "required_modalities"),
        comparison_queries=_profile_tuple(profile, "comparison_queries"),
        comparison_entities=_profile_tuple(profile, "comparison_entities"),
        has_comparison_intent=bool(getattr(profile, "has_comparison_intent", False)),
        has_aggregation_intent=bool(getattr(profile, "has_aggregation_intent", False)),
        has_temporal_intent=bool(getattr(profile, "has_temporal_intent", False)),
        inject_corpus_summaries=bool(getattr(profile, "inject_corpus_summaries", False)),
        terms=tuple(
            QueryTerm(
                text=str(item.get("text") or ""),
                origin=str(item.get("origin") or "unknown"),
            )
            for item in getattr(outcome, "query_terms", [])
            if isinstance(item, dict) and item.get("text")
        ),
    )
    collection = str(getattr(config, "collection", "default"))
    return RetrievalRun(
        run_id=str(uuid.uuid4()),
        created_at=_utc_now(),
        query=query,
        evidence=pack,
        strategies=_strategy_executions(getattr(outcome, "retrieval_trace", {})),
        candidate_stages=_candidate_stages(getattr(outcome, "retrieval_trace", {})),
        pyrrho=pyrrho_execution(decision, evidence_count=len(selected)),
        ranked_evidence=_freeze_evidence(getattr(compilation, "results", [])),
        pyrrho_evidence=_freeze_evidence(selected),
        environment=RunEnvironment(
            fitz_sage_version=_fitz_version(),
            engine="fitz_krag",
            collection=collection,
            config_sha256=_config_sha256(config),
            collection_sha256=_collection_sha256(
                workspace,
                collection,
                indexing_status,
            ),
            components=_component_specs(config),
            indexing_status=dict(indexing_status),
        ),
        warnings=(),
        content_included=True,
    )


def pyrrho_execution(decision: Any, *, evidence_count: int) -> PyrrhoExecution:
    """Record Pyrrho's public decision without reinterpretation."""
    payload = decision.to_dict()
    if not isinstance(payload, dict):
        raise TypeError("Pyrrho decision.to_dict() must return a dictionary.")
    verdict = getattr(decision, "verdict", None)
    if not isinstance(verdict, str) or not verdict:
        raise ValueError("Pyrrho decision is missing its verdict.")
    return PyrrhoExecution(
        verdict=verdict,
        evidence_count=evidence_count,
        reasons=tuple(str(reason) for reason in decision.reasons if reason),
        decision=payload,
    )


def _strategy_executions(trace: Any) -> tuple[StrategyExecution, ...]:
    executions: list[StrategyExecution] = []
    _append_strategy_executions(executions, trace, prefix="")
    closure = trace.get("evidence_closure") if isinstance(trace, dict) else None
    if isinstance(closure, dict):
        runs = closure.get("runs")
        if isinstance(runs, list):
            for index, run in enumerate(runs, start=1):
                if not isinstance(run, dict):
                    continue
                _append_strategy_executions(
                    executions,
                    run.get("trace"),
                    prefix=f"closure_{index}:",
                )
    return tuple(executions)


def _append_strategy_executions(
    output: list[StrategyExecution],
    trace: Any,
    *,
    prefix: str,
) -> None:
    if not isinstance(trace, dict):
        return
    router = trace.get("router")
    if not isinstance(router, dict):
        return
    calls = router.get("strategy_calls")
    if not isinstance(calls, list):
        calls = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        output.append(
            StrategyExecution(
                strategy=prefix + str(call.get("strategy") or "unknown"),
                query=str(call.get("query") or ""),
                result_count=_int(call.get("count")),
                succeeded=not bool(call.get("error")),
            )
        )
    for name, strategy_trace in (
        ("corpus_summary", router.get("corpus_summary")),
    ):
        if not isinstance(strategy_trace, dict) or not strategy_trace.get("enabled"):
            continue
        output.append(
            StrategyExecution(
                strategy=prefix + name,
                query=str(trace.get("query") or ""),
                result_count=_int(strategy_trace.get("count")),
                succeeded=not bool(strategy_trace.get("error")),
            )
        )


def _candidate_stages(trace: Any) -> tuple[CandidateStage, ...]:
    stages: list[CandidateStage] = []
    _append_candidate_stages(stages, trace, prefix="")
    closure = trace.get("evidence_closure") if isinstance(trace, dict) else None
    if isinstance(closure, dict):
        runs = closure.get("runs")
        if isinstance(runs, list):
            for index, run in enumerate(runs, start=1):
                if not isinstance(run, dict):
                    continue
                _append_candidate_stages(
                    stages,
                    run.get("trace"),
                    prefix=f"closure_{index}:",
                )
    return tuple(stages)


def _append_candidate_stages(
    output: list[CandidateStage],
    trace: Any,
    *,
    prefix: str,
) -> None:
    if not isinstance(trace, dict):
        return
    stage_sources: list[tuple[str, Any]] = [
        ("recall", trace.get("recall")),
        (
            "reranked",
            (
                (trace.get("reranker") or {}).get("output")
                if isinstance(trace.get("reranker"), dict)
                else None
            ),
        ),
        ("final", trace.get("final_addresses")),
    ]
    for name, raw_candidates in stage_sources:
        if not isinstance(raw_candidates, list):
            continue
        candidates = tuple(
            _candidate_reference(item, default_rank=index)
            for index, item in enumerate(raw_candidates, start=1)
            if isinstance(item, dict)
        )
        output.append(CandidateStage(name=prefix + name, candidates=candidates))


def _candidate_reference(
    raw: dict[str, Any],
    *,
    default_rank: int,
) -> CandidateReference:
    return CandidateReference(
        rank=max(1, _int(raw.get("rank"), default=default_rank)),
        kind=str(raw.get("kind") or ""),
        source_id=str(raw.get("source_id") or ""),
        location=str(raw.get("location") or ""),
        score=_optional_float(raw.get("score")),
    )


def _freeze_evidence(results: Any) -> tuple[FrozenEvidence, ...]:
    if not isinstance(results, list):
        return ()
    frozen: list[FrozenEvidence] = []
    for rank, result in enumerate(results, start=1):
        address = getattr(result, "address", None)
        metadata = getattr(result, "metadata", {}) or {}
        compiler = metadata.get("evidence_compiler") if isinstance(metadata, dict) else None
        kind = getattr(getattr(address, "kind", None), "value", "")
        frozen.append(
            FrozenEvidence.create(
                rank=rank,
                source_id=str(getattr(address, "source_id", "") or ""),
                file_path=str(getattr(result, "file_path", "") or ""),
                address_kind=str(kind),
                address_location=str(getattr(address, "location", "") or ""),
                address_summary=str(getattr(address, "summary", "") or ""),
                line_range=getattr(result, "line_range", None),
                score=_optional_float(getattr(address, "score", None)),
                content=str(getattr(result, "content", "") or ""),
                compiler_metadata=(dict(compiler) if isinstance(compiler, dict) else {}),
            )
        )
    return tuple(frozen)


def _component_specs(config: Any) -> dict[str, str]:
    from fitz_sage.llm.providers.onnx_chat import DEFAULT_QWEN_MODEL_ALIAS
    from fitz_sage.llm.providers.onnx_reranker import DEFAULT_MODEL_ID as DEFAULT_RERANKER_MODEL_ID

    reranker = str(getattr(config, "rerank", "unknown"))
    if reranker == "onnx":
        reranker = f"onnx/{DEFAULT_RERANKER_MODEL_ID}"
    output = {
        "semantic_keywords": f"onnx/{DEFAULT_QWEN_MODEL_ALIAS}",
        "reranker": reranker,
        "pyrrho": str(getattr(config, "governance", "unknown")),
    }
    for name in ("query_intelligence", "synthesizer", "parser", "vision"):
        value = getattr(config, name, None)
        output[name] = str(value) if value is not None else "disabled"
    return output


def _config_sha256(config: Any) -> str:
    raw = config.model_dump(mode="json") if hasattr(config, "model_dump") else vars(config)
    safe = {
        str(key): value
        for key, value in raw.items()
        if not any(part in str(key).casefold() for part in _SENSITIVE_CONFIG_PARTS)
    }
    payload = json.dumps(safe, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collection_sha256(
    workspace: Path,
    collection: str,
    indexing_status: dict[str, Any],
) -> str | None:
    manifest_path = workspace / "collections" / collection / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        digest = hashlib.sha256()
        digest.update(manifest_path.read_bytes())
        digest.update(
            json.dumps(
                indexing_status,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        return digest.hexdigest()
    except OSError:
        return None


def _fitz_version() -> str:
    try:
        from importlib.metadata import version

        return version("fitz-sage")
    except Exception:
        from fitz_sage import __version__

        return __version__


def _profile_tuple(profile: Any, name: str) -> tuple[str, ...]:
    value = getattr(profile, name, ())
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item) for item in value if item)


def _optional_profile_text(profile: Any, name: str) -> str | None:
    return _optional_text(getattr(profile, name, None))


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _query_shape(profile: Any) -> str:
    if bool(getattr(profile, "has_comparison_intent", False)):
        return "comparison"
    if bool(getattr(profile, "has_aggregation_intent", False)):
        return "broad"
    if bool(getattr(profile, "has_temporal_intent", False)):
        return "temporal"
    return "narrow"


__all__ = ["build_retrieval_run", "pyrrho_execution"]
