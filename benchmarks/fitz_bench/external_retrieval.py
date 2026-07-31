"""Shared execution machinery for labeled external retrieval benchmarks."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.retrieval_ablation import RetrievalAblation, apply_ablation
from benchmarks.fitz_bench.retrieval_eval import (
    aggregate_metrics,
    metric_delta,
    ranking_metrics,
    stage_failure,
    stage_recoveries,
    summarize_latency,
)
from benchmarks.fitz_bench.timing import group_timings, summarize_timing_records
from fitz_sage.config.loader import load_engine_config
from fitz_sage.core import Query
from fitz_sage.core.paths import FitzPaths
from fitz_sage.runtime import create_engine
from fitz_sage.storage.sqlite import SqliteConnectionManager

STAGES = ("baseline", "recall", "reranked", "final", "compiled", "delivered")


@dataclass(frozen=True)
class ExternalRetrievalDataset:
    """One corpus, query set, and identity contract for external evaluation."""

    name: str
    report: dict[str, Any]
    fingerprint: dict[str, Any]
    corpus_dir: Path
    mapping_path: Path
    corpus_documents: int
    adapter_schema_version: int
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    baseline_factory: Callable[[], Any]
    baseline_report: dict[str, Any]
    ignore_identical_ids: bool = False
    allow_duplicate_document_ids: bool = False


@dataclass(frozen=True)
class ExternalQuerySelection:
    """Frozen ordered query selection and per-query grouping metadata."""

    name: str
    query_ids: tuple[str, ...]
    metadata: dict[str, dict[str, Any]]
    report: dict[str, Any]
    digest: str | None = None


@dataclass(frozen=True)
class MappingIndex:
    """Validated external-document mapping keyed by projected source path."""

    by_path: dict[str, str]
    paths_by_document: dict[str, tuple[str, ...]]
    hashes_by_path: dict[str, str]


def evaluate_external_dataset(
    dataset: ExternalRetrievalDataset,
    selection: ExternalQuerySelection,
    *,
    workspace_root: Path,
    run_id: str,
    namespace: str,
    cutoffs: list[int],
    index_mode: str,
    governance: str | None,
    baseline_only: bool,
    reuse_workspace: bool,
    reuse_index: bool,
    resume_queries: bool,
    evaluation_paths: Sequence[Path],
    ablation: RetrievalAblation | None = None,
) -> dict[str, Any]:
    """Evaluate plain BM25 and Fitz-Sage over one frozen external selection."""
    _validate_selection(dataset, selection)
    query_ids = list(selection.query_ids)
    qrels = {query_id: dataset.qrels[query_id] for query_id in query_ids}
    max_k = max(cutoffs)

    baseline_started = time.perf_counter()
    baseline = dataset.baseline_factory()
    baseline_build_seconds = time.perf_counter() - baseline_started
    records: dict[str, dict[str, Any]] = {}
    baseline_durations: list[float] = []
    baseline_action = str(getattr(baseline, "action", "built"))
    try:
        for index, query_id in enumerate(query_ids, start=1):
            query_started = time.perf_counter()
            ranking = filter_ranking(
                baseline.search(dataset.queries[query_id], top_k=max_k),
                excluded_document_id=(query_id if dataset.ignore_identical_ids else None),
                limit=max_k,
            )
            duration = time.perf_counter() - query_started
            baseline_durations.append(duration)
            record = {
                "query_id": query_id,
                "query": dataset.queries[query_id],
                "judgments": qrels[query_id],
                "rankings": {"baseline": ranking},
                "metrics": {"baseline": ranking_metrics(ranking, qrels[query_id], cutoffs)},
                "latency_seconds": {"baseline": duration},
            }
            metadata = selection.metadata.get(query_id)
            if metadata is not None:
                record["evaluation"] = metadata
            records[query_id] = record
            if index % 100 == 0 or index == len(query_ids):
                print(
                    f"  BM25 {dataset.name}: {index}/{len(query_ids)} queries",
                    flush=True,
                )
    finally:
        close = getattr(baseline, "close", None)
        if callable(close):
            close()
        del baseline
        gc.collect()

    selection_report = {
        **selection.report,
        "name": selection.name,
        "queries": len(query_ids),
        "cutoffs": cutoffs,
        "ignore_identical_ids": dataset.ignore_identical_ids,
    }
    result: dict[str, Any] = {
        "dataset": dataset.report,
        "selection": selection_report,
        "baseline": {
            **dataset.baseline_report,
            "build_seconds": baseline_build_seconds,
            "action": baseline_action,
            "latency": summarize_latency(baseline_durations),
        },
    }
    if baseline_only:
        result["summary"] = summarize_records(records)
        result["records"] = list(records.values())
        return result

    workspace_key = dataset.name if reuse_workspace else f"{run_id}-{dataset.name}"
    workspace = (Path(workspace_root).resolve() / workspace_key).resolve()
    collection = f"{namespace}_{dataset.name}_v{dataset.adapter_schema_version}_{index_mode}"
    activate_workspace(workspace)
    engine = create_benchmark_engine(collection, governance=governance, ablation=ablation)
    engine.load(collection)
    manifest = None
    index_action = "indexed"
    indexing_started = time.perf_counter()
    try:
        if reuse_index and persisted_index_exists(workspace, collection):
            manifest = require_reusable_index(
                engine,
                expected_source=dataset.corpus_dir,
                expected_documents=dataset.corpus_documents,
                index_mode=index_mode,
            )
            index_action = "reused_verified"
        else:
            manifest = engine.point(
                dataset.corpus_dir,
                collection=collection,
                start_worker=False,
            )
            if index_mode == "complete":
                engine.continue_enrichment()
        indexing_status = dict(engine.indexing_status())
        source_ids, mapping_summary = source_id_mapping(
            manifest,
            dataset.mapping_path,
            allow_duplicate_document_ids=dataset.allow_duplicate_document_ids,
        )
        indexing_seconds = time.perf_counter() - indexing_started
        checkpoint_path = checkpoint_path_for(
            workspace,
            namespace=namespace,
            ablation=ablation,
            selection_digest=selection.digest,
        )
        checkpoint_signature = {
            "schema_version": 1,
            "namespace": namespace,
            "dataset": dataset.name,
            "dataset_fingerprint": dataset.fingerprint,
            "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode("utf-8")).hexdigest(),
            "queries": len(query_ids),
            "cutoffs": cutoffs,
            "index_mode": index_mode,
            "governance": governance,
            "ablation": ablation.as_dict() if ablation else None,
            "selection_digest": selection.digest,
            "evaluation_source_sha256": evaluation_source_digest(evaluation_paths),
        }
        checkpoint_records = (
            load_checkpoint(checkpoint_path, checkpoint_signature) if resume_queries else {}
        )
        if not checkpoint_records:
            initialize_checkpoint(checkpoint_path, checkpoint_signature)

        fitz_durations: list[float] = []
        for index, query_id in enumerate(query_ids, start=1):
            if query_id in checkpoint_records:
                records[query_id] = checkpoint_records[query_id]
                fitz_durations.append(float(records[query_id]["latency_seconds"]["fitz_sage"]))
                if index % 25 == 0 or index == len(query_ids):
                    print(
                        f"  Fitz-Sage {dataset.name}: {index}/{len(query_ids)} queries "
                        "(resumed)",
                        flush=True,
                    )
                continue
            query_started = time.perf_counter()
            run = engine.trace(Query(text=dataset.queries[query_id]), top_k=max_k)
            duration = time.perf_counter() - query_started
            fitz_durations.append(duration)
            rankings, unmapped = run_rankings(
                run,
                source_ids,
                excluded_document_id=(query_id if dataset.ignore_identical_ids else None),
                limit=max_k,
            )
            query_record: dict[str, Any] = records[query_id]
            query_record["rankings"].update(rankings)
            query_record["metrics"].update(
                {
                    stage: ranking_metrics(ranking, qrels[query_id], cutoffs)
                    for stage, ranking in rankings.items()
                }
            )
            query_record["latency_seconds"]["fitz_sage"] = duration
            stage_seconds = {
                str(name): float(stage_duration)
                for name, stage_duration in run.evidence.timings.items()
            }
            grouped_seconds, timing_overlap = group_timings(
                stage_seconds,
                total_seconds=duration,
            )
            query_record["timing"] = {
                "total_seconds": duration,
                "stage_seconds": stage_seconds,
                "grouped_seconds": grouped_seconds,
                "timing_overlap_seconds": timing_overlap,
            }
            query_record["failure_attribution"] = stage_failure(rankings, qrels[query_id])
            query_record["recoveries"] = stage_recoveries(rankings, qrels[query_id])
            query_record["query_execution"] = run.query.to_dict()
            query_record["pyrrho"] = run.pyrrho.to_dict()
            query_record["unmapped_candidates"] = unmapped
            append_checkpoint(checkpoint_path, query_record)
            if index % 25 == 0 or index == len(query_ids):
                print(f"  Fitz-Sage {dataset.name}: {index}/{len(query_ids)} queries", flush=True)
    finally:
        engine.stop_background_enrichment()

    summary = summarize_records(records)
    summary["deltas_vs_plain_bm25"] = {
        stage: metric_delta(summary["metrics"][stage], summary["metrics"]["baseline"])
        for stage in STAGES
        if stage != "baseline" and stage in summary["metrics"]
    }
    result.update(
        {
            "workspace": str(workspace),
            "collection": collection,
            "index_mode": index_mode,
            "ingestion": {
                "duration_seconds": indexing_seconds,
                "action": index_action,
                "status": indexing_status,
                "mapping": mapping_summary,
                "failures": manifest_failures(manifest),
            },
            "fitz_sage": {
                "latency": summarize_latency(fitz_durations),
                "governance_override": governance,
                "ablation": ablation.as_dict() if ablation else None,
                "query_checkpoint": str(checkpoint_path),
                "resumed_queries": len(checkpoint_records),
            },
            "summary": summary,
            "records": list(records.values()),
        }
    )
    return result


def load_mapping(
    path: Path,
    *,
    allow_duplicate_document_ids: bool = False,
) -> MappingIndex:
    """Load a path-to-external-ID mapping with optional duplicate IDs."""
    by_path: dict[str, str] = {}
    paths_by_document: dict[str, list[str]] = {}
    hashes_by_path: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid mapping JSONL at {path}:{line_number}") from exc
            if not isinstance(raw, dict):
                raise TypeError(f"Expected mapping object at {path}:{line_number}")
            document_id = _required_mapping_string(raw, "document_id", path, line_number)
            relative_path = _required_mapping_string(
                raw, "relative_path", path, line_number
            ).replace("\\", "/")
            content_sha256 = _required_mapping_string(raw, "content_sha256", path, line_number)
            if relative_path in by_path:
                raise ValueError(f"Duplicate mapping path: {relative_path}")
            existing = paths_by_document.get(document_id)
            if existing and not allow_duplicate_document_ids:
                raise ValueError(f"Duplicate mapping document ID: {document_id}")
            by_path[relative_path] = document_id
            paths_by_document.setdefault(document_id, []).append(relative_path)
            hashes_by_path[relative_path] = content_sha256
    return MappingIndex(
        by_path=by_path,
        paths_by_document={key: tuple(value) for key, value in paths_by_document.items()},
        hashes_by_path=hashes_by_path,
    )


def run_rankings(
    run: Any,
    source_ids: dict[str, str],
    *,
    excluded_document_id: str | None = None,
    limit: int | None = None,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Map Fitz-Sage candidate stages back to external document IDs."""
    rankings: dict[str, list[str]] = {}
    unmapped: dict[str, int] = {}
    for stage in run.candidate_stages:
        if stage.name not in {"recall", "reranked", "final"}:
            continue
        values, missing = _map_source_ids(
            (candidate.source_id for candidate in stage.candidates),
            source_ids,
        )
        rankings[stage.name] = filter_ranking(
            values,
            excluded_document_id=excluded_document_id,
            limit=limit,
        )
        unmapped[stage.name] = missing
    for name, evidence in (
        ("compiled", run.ranked_evidence),
        ("delivered", run.pyrrho_evidence),
    ):
        values, missing = _map_source_ids(
            (item.source_id for item in evidence),
            source_ids,
        )
        rankings[name] = filter_ranking(
            values,
            excluded_document_id=excluded_document_id,
            limit=limit,
        )
        unmapped[name] = missing
    for name in ("recall", "reranked", "final", "compiled", "delivered"):
        rankings.setdefault(name, [])
        unmapped.setdefault(name, 0)
    return rankings, unmapped


def filter_ranking(
    ranking: list[str],
    *,
    excluded_document_id: str | None,
    limit: int | None,
) -> list[str]:
    """Remove an optional self-document and enforce a stable output depth."""
    filtered = [
        document_id
        for document_id in ranking
        if excluded_document_id is None or document_id != excluded_document_id
    ]
    return filtered[:limit] if limit is not None else filtered


def source_id_mapping(
    manifest: Any,
    mapping_path: Path,
    *,
    allow_duplicate_document_ids: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Join Fitz manifest source IDs to verified external document IDs."""
    mapping = load_mapping(
        mapping_path,
        allow_duplicate_document_ids=allow_duplicate_document_ids,
    )
    entries = manifest.entries()
    source_ids: dict[str, str] = {}
    missing_paths: list[str] = []
    hash_mismatches: list[str] = []
    mapped_paths: set[str] = set()
    for relative_path, entry in entries.items():
        normalized = str(relative_path).replace("\\", "/")
        document_id = mapping.by_path.get(normalized)
        if document_id is None:
            missing_paths.append(normalized)
            continue
        if str(entry.content_hash) != mapping.hashes_by_path[normalized]:
            hash_mismatches.append(normalized)
            continue
        mapped_paths.add(normalized)
        source_ids[str(entry.file_id)] = document_id
    unmapped_paths = sorted(set(mapping.by_path) - mapped_paths)
    if missing_paths or hash_mismatches or unmapped_paths:
        raise ValueError(
            "External adapter/manifest identity mismatch: "
            f"manifest_paths={missing_paths[:3]} "
            f"content_hashes={hash_mismatches[:3]} "
            f"adapter_paths={unmapped_paths[:3]}"
        )
    return source_ids, {
        "manifest_entries": len(entries),
        "adapter_files": len(mapping.by_path),
        "unique_document_ids": len(mapping.paths_by_document),
        "duplicate_document_ids": sum(
            len(paths) > 1 for paths in mapping.paths_by_document.values()
        ),
        "mapped_source_ids": len(source_ids),
        "verified_content_hashes": len(mapped_paths),
        "complete": len(mapped_paths) == len(mapping.by_path),
    }


def persisted_index_exists(workspace: Path, collection: str) -> bool:
    """Return whether a collection has a persisted source manifest."""
    return (workspace / "collections" / collection / "manifest.json").is_file()


def require_reusable_index(
    engine: Any,
    *,
    expected_source: Path,
    expected_documents: int,
    index_mode: str,
) -> Any:
    """Require an exact query-ready persisted index without rescanning files."""
    manifest = getattr(engine, "_manifest", None)
    source_dir = getattr(engine, "_source_dir", None)
    if manifest is None or source_dir is None:
        raise FileNotFoundError("Reusable collection has no persisted source manifest.")
    if Path(source_dir).resolve() != Path(expected_source).resolve():
        raise ValueError(
            "Reusable collection points at a different source directory: "
            f"{Path(source_dir).resolve()} != {Path(expected_source).resolve()}"
        )
    status = dict(engine.indexing_status())
    if (
        not status.get("query_ready")
        or int(status.get("indexed", 0)) != expected_documents
        or int(status.get("failed", 0))
        or int(status.get("unsupported", 0))
    ):
        raise ValueError(
            "Reusable collection is not a complete source index: "
            f"expected={expected_documents} status={status}"
        )
    enrichment = status.get("enrichment", {})
    if index_mode == "complete" and not (
        isinstance(enrichment, dict) and enrichment.get("complete")
    ):
        raise ValueError("Reusable collection has incomplete enrichment.")
    return manifest


def summarize_records(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate quality, stage, timing, and exact Pyrrho observations."""
    values = list(records.values())
    stages = sorted(
        {stage for record in values for stage in record.get("metrics", {})},
        key=lambda stage: STAGES.index(stage),
    )
    metrics = {
        stage: aggregate_metrics(
            record["metrics"][stage] for record in values if stage in record.get("metrics", {})
        )
        for stage in stages
    }
    failure_attribution = Counter(
        str(record["failure_attribution"]) for record in values if "failure_attribution" in record
    )
    recoveries = Counter(recovery for record in values for recovery in record.get("recoveries", []))
    pyrrho = Counter(str(record["pyrrho"]["verdict"]) for record in values if "pyrrho" in record)
    stage_retrieval = {stage: stage_retrieval_summary(values, stage) for stage in stages}
    timing = summarize_timing_records(
        [record["timing"] for record in values if isinstance(record.get("timing"), dict)]
    )
    return {
        "queries": len(values),
        "metrics": metrics,
        "stage_retrieval": stage_retrieval,
        "failure_attribution": dict(sorted(failure_attribution.items())),
        "recoveries": dict(sorted(recoveries.items())),
        "pyrrho_verdicts": dict(sorted(pyrrho.items())),
        "timing": timing,
    }


def stage_retrieval_summary(
    records: list[dict[str, Any]],
    stage: str,
) -> dict[str, float]:
    """Summarize unique depth and relevant-hit coverage for one stage."""
    rankings = [
        record["rankings"][stage] for record in records if stage in record.get("rankings", {})
    ]
    if not rankings:
        return {"mean_unique_documents": 0.0, "relevant_hit_rate": 0.0}
    hits = 0
    for record in records:
        ranking = record.get("rankings", {}).get(stage)
        if ranking is None:
            continue
        relevant = {document_id for document_id, score in record["judgments"].items() if score > 0}
        hits += bool(relevant.intersection(ranking))
    return {
        "mean_unique_documents": sum(len(ranking) for ranking in rankings) / len(rankings),
        "relevant_hit_rate": hits / len(rankings),
    }


def manifest_failures(manifest: Any) -> list[dict[str, str]]:
    """Return explicit failed and unsupported manifest records."""
    return [
        {
            "path": entry.rel_path,
            "state": entry.state.value,
            "stage": entry.failure_stage or "",
            "message": entry.failure_message or "",
        }
        for entry in manifest.entries().values()
        if entry.state.value in {"failed", "unsupported"}
    ]


def checkpoint_path_for(
    workspace: Path,
    *,
    namespace: str,
    ablation: RetrievalAblation | None,
    selection_digest: str | None,
) -> Path:
    """Return a variant- and selection-specific checkpoint path."""
    suffix = f"-{ablation.name}" if ablation is not None else ""
    if selection_digest is not None:
        suffix += f"-{selection_digest[:12]}"
    return workspace / f"{namespace}-query-checkpoint{suffix}.jsonl"


def initialize_checkpoint(path: Path, signature: dict[str, Any]) -> None:
    """Replace a checkpoint with one exact signature header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({"type": "header", "signature": signature}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    """Durably append one completed query record."""
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps({"type": "query", "record": record}) + "\n")


def load_checkpoint(
    path: Path,
    expected_signature: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load matching complete records, tolerating only a truncated final line."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid external benchmark checkpoint header: {path}") from exc
    if (
        not isinstance(header, dict)
        or header.get("type") != "header"
        or header.get("signature") != expected_signature
    ):
        raise ValueError(
            f"External benchmark checkpoint does not match this evaluation; "
            f"remove or replace {path}"
        )
    records: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines[1:], start=2):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines):
                break
            raise ValueError(f"Invalid checkpoint record at {path}:{index}")
        if not isinstance(item, dict) or item.get("type") != "query":
            raise ValueError(f"Invalid checkpoint record at {path}:{index}")
        record = item.get("record")
        if not isinstance(record, dict) or not isinstance(record.get("query_id"), str):
            raise TypeError(f"Invalid checkpoint query at {path}:{index}")
        records[record["query_id"]] = record
    return records


def activate_workspace(workspace: Path) -> None:
    """Activate one isolated benchmark workspace."""
    workspace.mkdir(parents=True, exist_ok=True)
    SqliteConnectionManager.reset_instance()
    FitzPaths.set_workspace(workspace)


def create_benchmark_engine(
    collection: str,
    *,
    governance: str | None,
    ablation: RetrievalAblation | None = None,
) -> Any:
    """Create the canonical engine with an optional benchmark-only ablation."""
    config = load_engine_config("fitz_krag")
    values = config.model_dump()
    values["collection"] = collection
    if governance is not None:
        values["governance"] = governance
    engine = create_engine("fitz_krag", config=type(config)(**values))
    if ablation is not None:
        apply_ablation(engine, ablation)
    return engine


def evaluation_source_digest(extra_paths: Sequence[Path]) -> str:
    """Hash package and benchmark sources that can change measured behavior."""
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "fitz_sage").rglob("*.py"))
    paths.extend((root / "fitz_sage").rglob("*.yaml"))
    paths.append(Path(__file__).resolve())
    paths.extend(Path(path).resolve() for path in extra_paths)
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_selection(
    dataset: ExternalRetrievalDataset,
    selection: ExternalQuerySelection,
) -> None:
    if not selection.query_ids:
        raise ValueError("External retrieval selection has no queries.")
    if len(set(selection.query_ids)) != len(selection.query_ids):
        raise ValueError("External retrieval selection contains duplicate query IDs.")
    missing_queries = [
        query_id for query_id in selection.query_ids if query_id not in dataset.queries
    ]
    missing_qrels = [query_id for query_id in selection.query_ids if query_id not in dataset.qrels]
    if missing_queries or missing_qrels:
        raise ValueError(
            "External retrieval selection references missing data: "
            f"queries={missing_queries[:3]} qrels={missing_qrels[:3]}"
        )
    without_positive = [
        query_id
        for query_id in selection.query_ids
        if not any(score > 0 for score in dataset.qrels[query_id].values())
    ]
    if without_positive:
        raise ValueError(
            "Scored external retrieval queries require positive documents: "
            f"{without_positive[:3]}"
        )


def _map_source_ids(
    values: Any,
    source_ids: dict[str, str],
) -> tuple[list[str], int]:
    documents: list[str] = []
    seen: set[str] = set()
    missing = 0
    for source_id in values:
        document_id = source_ids.get(str(source_id))
        if document_id is None:
            missing += 1
            continue
        if document_id not in seen:
            seen.add(document_id)
            documents.append(document_id)
    return documents, missing


def _required_mapping_string(
    raw: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Mapping field {key!r} is invalid at {path}:{line_number}")
    return value
