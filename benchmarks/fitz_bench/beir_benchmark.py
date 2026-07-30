"""Run labeled BEIR retrieval evaluation through Fitz-Sage's canonical path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.beir import (
    DATASETS,
    PreparedDataset,
    iter_corpus,
    load_mapping,
    load_qrels,
    load_queries,
    prepare_dataset,
    projected_content,
)
from benchmarks.fitz_bench.retrieval_ablation import (
    RetrievalAblation,
    ablation_names,
    apply_ablation,
    get_ablation,
)
from benchmarks.fitz_bench.retrieval_eval import (
    PlainBm25,
    aggregate_metrics,
    metric_delta,
    metric_formulas,
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

_STAGES = ("baseline", "recall", "reranked", "final", "compiled", "delivered")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    ablation = get_ablation(args.ablation) if args.ablation else None
    datasets = args.datasets or list(DATASETS)
    prepared = [
        prepare_dataset(
            args.cache_dir,
            name,
            max_download_bytes=int(args.max_download_gib * 1024**3),
            max_extracted_bytes=int(args.max_extracted_gib * 1024**3),
            offline=args.offline,
            progress=print,
        )
        for name in datasets
    ]

    if args.download_only:
        report = {
            "run": _run_metadata(args, run_id, started),
            "datasets": [{"dataset": item.as_dict()} for item in prepared],
            "gate": {"type": "operational_only", "passed": True},
        }
        return _write_report(report, args.output, args.markdown)

    results: list[dict[str, Any]] = []
    for index, dataset in enumerate(prepared, start=1):
        print(f"[{index}/{len(prepared)}] Evaluating BEIR {dataset.name}", flush=True)
        result = evaluate_dataset(
            dataset,
            workspace_root=args.workspace_root,
            run_id=run_id,
            cutoffs=args.cutoffs,
            query_limit=args.query_limit,
            index_mode=args.index_mode,
            governance=args.governance,
            baseline_only=args.baseline_only,
            reuse_workspace=args.reuse_workspace,
            resume_queries=args.resume_queries,
            ablation=ablation,
        )
        results.append(result)
        _write_partial_report(
            args.output,
            {
                "run": _run_metadata(args, run_id, started),
                "metric_formulas": metric_formulas(),
                "datasets": results,
                "complete": len(results) == len(prepared),
            },
        )

    gate = _gate(results, max_recall_regression=args.max_recall_regression)
    report = {
        "run": _run_metadata(args, run_id, started),
        "metric_formulas": metric_formulas(),
        "datasets": results,
        "gate": gate,
        "complete": True,
    }
    exit_code = _write_report(report, args.output, args.markdown)
    return exit_code if gate["passed"] else 1


def evaluate_dataset(
    dataset: PreparedDataset,
    *,
    workspace_root: Path,
    run_id: str,
    cutoffs: list[int],
    query_limit: int | None,
    index_mode: str,
    governance: str | None,
    baseline_only: bool,
    reuse_workspace: bool,
    resume_queries: bool,
    ablation: RetrievalAblation | None = None,
) -> dict[str, Any]:
    """Evaluate one prepared dataset with plain BM25 and Fitz-Sage."""
    queries = load_queries(Path(dataset.source_queries))
    all_qrels = load_qrels(Path(dataset.source_qrels))
    query_ids = list(all_qrels)
    if query_limit is not None:
        query_ids = query_ids[:query_limit]
    qrels = {query_id: all_qrels[query_id] for query_id in query_ids}
    max_k = max(cutoffs)

    baseline_started = time.perf_counter()
    baseline = PlainBm25.build(
        (str(record["_id"]), projected_content(record))
        for record in iter_corpus(Path(dataset.source_corpus))
    )
    baseline_build_seconds = time.perf_counter() - baseline_started
    records: dict[str, dict[str, Any]] = {}
    baseline_durations: list[float] = []
    for index, query_id in enumerate(query_ids, start=1):
        query_started = time.perf_counter()
        ranking = baseline.search(queries[query_id], top_k=max_k)
        duration = time.perf_counter() - query_started
        baseline_durations.append(duration)
        records[query_id] = {
            "query_id": query_id,
            "query": queries[query_id],
            "judgments": qrels[query_id],
            "rankings": {"baseline": ranking},
            "metrics": {
                "baseline": ranking_metrics(ranking, qrels[query_id], cutoffs),
            },
            "latency_seconds": {"baseline": duration},
        }
        if index % 100 == 0 or index == len(query_ids):
            print(
                f"  BM25 {dataset.name}: {index}/{len(query_ids)} queries",
                flush=True,
            )
    del baseline
    gc.collect()

    result: dict[str, Any] = {
        "dataset": dataset.as_dict(),
        "selection": {
            "split": "test",
            "queries": len(query_ids),
            "query_limit": query_limit,
            "cutoffs": cutoffs,
        },
        "baseline": {
            "implementation": "benchmark-local Okapi BM25",
            "analyzer": "Unicode word tokens, casefolded",
            "k1": 1.2,
            "b": 0.75,
            "build_seconds": baseline_build_seconds,
            "latency": summarize_latency(baseline_durations),
        },
    }
    if baseline_only:
        result["summary"] = _summarize_records(records)
        result["records"] = list(records.values())
        return result

    workspace_key = dataset.name if reuse_workspace else f"{run_id}-{dataset.name}"
    workspace = (Path(workspace_root).resolve() / workspace_key).resolve()
    collection = f"beir_{dataset.name}_v{dataset.adapter_schema_version}_{index_mode}"
    _activate_workspace(workspace)
    engine = _create_engine(collection, governance=governance, ablation=ablation)
    engine.load(collection)
    manifest = None
    indexing_started = time.perf_counter()
    try:
        manifest = engine.point(
            Path(dataset.corpus_dir),
            collection=collection,
            start_worker=False,
        )
        if index_mode == "complete":
            engine.continue_enrichment()
        indexing_seconds = time.perf_counter() - indexing_started
        indexing_status = dict(engine.indexing_status())
        source_ids, mapping_summary = _source_id_mapping(
            manifest,
            Path(dataset.mapping_path),
        )
        checkpoint_path = _checkpoint_path(workspace, ablation)
        checkpoint_signature = _checkpoint_signature(
            dataset,
            query_ids=query_ids,
            cutoffs=cutoffs,
            index_mode=index_mode,
            governance=governance,
            ablation=ablation,
        )
        checkpoint_records = (
            _load_checkpoint(checkpoint_path, checkpoint_signature) if resume_queries else {}
        )
        if not checkpoint_records:
            _initialize_checkpoint(checkpoint_path, checkpoint_signature)

        fitz_durations: list[float] = []
        for index, query_id in enumerate(query_ids, start=1):
            if query_id in checkpoint_records:
                records[query_id] = checkpoint_records[query_id]
                fitz_durations.append(float(records[query_id]["latency_seconds"]["fitz_sage"]))
                if index % 25 == 0 or index == len(query_ids):
                    print(
                        f"  Fitz-Sage {dataset.name}: {index}/{len(query_ids)} queries (resumed)",
                        flush=True,
                    )
                continue
            query_started = time.perf_counter()
            run = engine.trace(Query(text=queries[query_id]), top_k=max_k)
            duration = time.perf_counter() - query_started
            fitz_durations.append(duration)
            rankings, unmapped = _run_rankings(run, source_ids)
            record = records[query_id]
            record["rankings"].update(rankings)
            record["metrics"].update(
                {
                    stage: ranking_metrics(ranking, qrels[query_id], cutoffs)
                    for stage, ranking in rankings.items()
                }
            )
            record["latency_seconds"]["fitz_sage"] = duration
            stage_seconds = {
                str(name): float(stage_duration)
                for name, stage_duration in run.evidence.timings.items()
            }
            grouped_seconds, timing_overlap = group_timings(
                stage_seconds,
                total_seconds=duration,
            )
            record["timing"] = {
                "total_seconds": duration,
                "stage_seconds": stage_seconds,
                "grouped_seconds": grouped_seconds,
                "timing_overlap_seconds": timing_overlap,
            }
            record["failure_attribution"] = stage_failure(rankings, qrels[query_id])
            record["recoveries"] = stage_recoveries(rankings, qrels[query_id])
            record["query_execution"] = run.query.to_dict()
            record["pyrrho"] = run.pyrrho.to_dict()
            record["unmapped_candidates"] = unmapped
            _append_checkpoint(checkpoint_path, record)
            if index % 25 == 0 or index == len(query_ids):
                print(
                    f"  Fitz-Sage {dataset.name}: {index}/{len(query_ids)} queries",
                    flush=True,
                )
    finally:
        engine.stop_background_enrichment()

    summary = _summarize_records(records)
    summary["deltas_vs_plain_bm25"] = {
        stage: metric_delta(summary["metrics"][stage], summary["metrics"]["baseline"])
        for stage in _STAGES
        if stage not in {"baseline"} and stage in summary["metrics"]
    }
    result.update(
        {
            "workspace": str(workspace),
            "collection": collection,
            "index_mode": index_mode,
            "ingestion": {
                "duration_seconds": indexing_seconds,
                "status": indexing_status,
                "mapping": mapping_summary,
                "failures": _manifest_failures(manifest),
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


def _run_rankings(
    run: Any,
    source_ids: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    rankings: dict[str, list[str]] = {}
    unmapped: dict[str, int] = {}
    for stage in run.candidate_stages:
        if stage.name not in {"recall", "reranked", "final"}:
            continue
        values, missing = _map_source_ids(
            (candidate.source_id for candidate in stage.candidates),
            source_ids,
        )
        rankings[stage.name] = values
        unmapped[stage.name] = missing
    for name, evidence in (
        ("compiled", run.ranked_evidence),
        ("delivered", run.pyrrho_evidence),
    ):
        values, missing = _map_source_ids(
            (item.source_id for item in evidence),
            source_ids,
        )
        rankings[name] = values
        unmapped[name] = missing
    for name in ("recall", "reranked", "final", "compiled", "delivered"):
        rankings.setdefault(name, [])
        unmapped.setdefault(name, 0)
    return rankings, unmapped


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


def _source_id_mapping(
    manifest: Any,
    mapping_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    by_path, by_document = load_mapping(mapping_path)
    entries = manifest.entries()
    source_ids: dict[str, str] = {}
    missing_paths: list[str] = []
    for relative_path, entry in entries.items():
        normalized = str(relative_path).replace("\\", "/")
        document_id = by_path.get(normalized)
        if document_id is None:
            missing_paths.append(normalized)
            continue
        source_ids[str(entry.file_id)] = document_id
    unmapped_documents = sorted(set(by_document) - set(source_ids.values()))
    if missing_paths or unmapped_documents:
        raise ValueError(
            "BEIR adapter/manifest identity mismatch: "
            f"manifest_paths={missing_paths[:3]} documents={unmapped_documents[:3]}"
        )
    return source_ids, {
        "manifest_entries": len(entries),
        "adapter_documents": len(by_document),
        "mapped_source_ids": len(source_ids),
        "complete": len(source_ids) == len(by_document),
    }


def _summarize_records(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(records.values())
    stages = sorted(
        {stage for record in values for stage in record.get("metrics", {})},
        key=lambda stage: _STAGES.index(stage),
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
    stage_retrieval = {stage: _stage_retrieval_summary(values, stage) for stage in stages}
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


def _stage_retrieval_summary(
    records: list[dict[str, Any]],
    stage: str,
) -> dict[str, float]:
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


def _manifest_failures(manifest: Any) -> list[dict[str, str]]:
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


def _checkpoint_path(
    workspace: Path,
    ablation: RetrievalAblation | None,
) -> Path:
    suffix = f"-{ablation.name}" if ablation is not None else ""
    return workspace / f"beir-query-checkpoint{suffix}.jsonl"


def _checkpoint_signature(
    dataset: PreparedDataset,
    *,
    query_ids: list[str],
    cutoffs: list[int],
    index_mode: str,
    governance: str | None,
    ablation: RetrievalAblation | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset": dataset.name,
        "dataset_md5": dataset.md5,
        "adapter_schema_version": dataset.adapter_schema_version,
        "query_ids_sha256": hashlib.sha256("\n".join(query_ids).encode("utf-8")).hexdigest(),
        "queries": len(query_ids),
        "cutoffs": cutoffs,
        "index_mode": index_mode,
        "governance": governance,
        "ablation": ablation.as_dict() if ablation else None,
        "evaluation_source_sha256": _evaluation_source_digest(),
    }


def _evaluation_source_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    paths = list((root / "fitz_sage").rglob("*.py"))
    paths.extend((root / "fitz_sage").rglob("*.yaml"))
    paths.extend(
        root / "benchmarks" / "fitz_bench" / name
        for name in (
            "beir.py",
            "beir_benchmark.py",
            "retrieval_ablation.py",
            "retrieval_eval.py",
            "timing.py",
        )
    )
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _initialize_checkpoint(path: Path, signature: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({"type": "header", "signature": signature}) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps({"type": "query", "record": record}) + "\n")


def _load_checkpoint(
    path: Path,
    expected_signature: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return {}
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid BEIR checkpoint header: {path}") from exc
    if (
        not isinstance(header, dict)
        or header.get("type") != "header"
        or header.get("signature") != expected_signature
    ):
        raise ValueError(
            f"BEIR checkpoint does not match this evaluation; remove or replace {path}"
        )
    records: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines[1:], start=2):
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines):
                break
            raise ValueError(f"Invalid BEIR checkpoint record at {path}:{index}")
        if not isinstance(item, dict) or item.get("type") != "query":
            raise ValueError(f"Invalid BEIR checkpoint record at {path}:{index}")
        record = item.get("record")
        if not isinstance(record, dict) or not isinstance(record.get("query_id"), str):
            raise TypeError(f"Invalid BEIR checkpoint query at {path}:{index}")
        records[record["query_id"]] = record
    return records


def _activate_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    SqliteConnectionManager.reset_instance()
    FitzPaths.set_workspace(workspace)


def _create_engine(
    collection: str,
    *,
    governance: str | None,
    ablation: RetrievalAblation | None = None,
) -> Any:
    config = load_engine_config("fitz_krag")
    values = config.model_dump()
    values["collection"] = collection
    if governance is not None:
        values["governance"] = governance
    engine = create_engine("fitz_krag", config=type(config)(**values))
    if ablation is not None:
        apply_ablation(engine, ablation)
    return engine


def _gate(
    results: list[dict[str, Any]],
    *,
    max_recall_regression: float | None,
) -> dict[str, Any]:
    operational_failures: list[str] = []
    score_failures: list[str] = []
    for result in results:
        dataset = result["dataset"]["name"]
        ingestion = result.get("ingestion")
        if ingestion and ingestion["failures"]:
            operational_failures.append(f"{dataset}: ingestion failures")
        if max_recall_regression is None or "summary" not in result:
            continue
        metrics = result["summary"]["metrics"]
        if "recall" not in metrics:
            continue
        cutoff = max(result["selection"]["cutoffs"])
        key = f"Recall@{cutoff}"
        delta = metrics["recall"][key] - metrics["baseline"][key]
        if delta < -max_recall_regression:
            score_failures.append(
                f"{dataset}: recall delta {delta:.6f} below {-max_recall_regression:.6f}"
            )
    return {
        "type": (
            "operational_and_recall_regression"
            if max_recall_regression is not None
            else "operational_only"
        ),
        "max_recall_regression": max_recall_regression,
        "operational_failures": operational_failures,
        "score_failures": score_failures,
        "passed": not operational_failures and not score_failures,
    }


def _run_metadata(
    args: argparse.Namespace,
    run_id: str,
    started: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "git": _git_state(),
        "datasets": args.datasets or list(DATASETS),
        "index_mode": args.index_mode,
        "baseline_only": args.baseline_only,
        "query_limit": args.query_limit,
        "cutoffs": args.cutoffs,
        "governance_override": args.governance,
        "ablation": get_ablation(args.ablation).as_dict() if args.ablation else None,
        "reuse_workspace": args.reuse_workspace,
        "resume_queries": args.resume_queries,
        "duration_seconds": time.perf_counter() - started,
    }


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": command("rev-parse", "HEAD") or None,
        "branch": command("branch", "--show-current") or None,
        "dirty": bool(command("status", "--porcelain")),
    }


def _write_report(
    report: dict[str, Any],
    output: Path,
    markdown: Path | None,
) -> int:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown = Path(markdown).resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "gate": report["gate"]}, indent=2))
    return 0


def _write_partial_report(path: Path, report: dict[str, Any]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _markdown(report: dict[str, Any]) -> str:
    ablation = report["run"].get("ablation")
    ablation_name = ablation.get("name") if isinstance(ablation, dict) else "canonical"
    lines = [
        "# BEIR Retrieval Benchmark",
        "",
        f"- Run: `{report['run']['run_id']}`",
        f"- Gate: `{'PASS' if report['gate']['passed'] else 'FAIL'}`",
        f"- Index mode: `{report['run']['index_mode']}`",
        f"- Ablation: `{ablation_name}`",
        "",
    ]
    for result in report["datasets"]:
        dataset = result["dataset"]
        lines.extend(
            [
                f"## {dataset['name']}",
                "",
                (
                    f"{result.get('selection', {}).get('queries', 0)} judged test queries, "
                    f"{dataset['corpus_documents']} documents."
                ),
                "",
            ]
        )
        summary = result.get("summary", {})
        metrics = summary.get("metrics", {})
        if metrics:
            cutoffs = result["selection"]["cutoffs"]
            ranking_cutoff = max((value for value in cutoffs if value <= 10), default=min(cutoffs))
            selected_metrics = [
                f"NDCG@{ranking_cutoff}",
                f"Recall@{max(cutoffs)}",
                f"MRR@{ranking_cutoff}",
            ]
            lines.append("| Stage | " + " | ".join(selected_metrics) + " |")
            lines.append("| --- | " + " | ".join("---:" for _ in selected_metrics) + " |")
            for stage, values in metrics.items():
                cells = [f"{values.get(name, 0.0):.4f}" for name in selected_metrics]
                lines.append(f"| {stage} | " + " | ".join(cells) + " |")
            lines.append("")
        attribution = summary.get("failure_attribution", {})
        if attribution:
            lines.append(
                "Failure attribution: "
                + ", ".join(f"`{key}`={value}" for key, value in attribution.items())
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to evaluate. Repeat; defaults to all supported datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".benchmark-data/beir"),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(".bench_workspace/beir"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/beir_latest.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("benchmarks/results/beir_latest.md"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument(
        "--cutoff",
        dest="cutoffs",
        action="append",
        type=int,
        help="Metric cutoff. Repeat; defaults to 1, 3, 5, 10, 20, and 50.",
    )
    parser.add_argument("--index-mode", choices=("source", "complete"), default="source")
    parser.add_argument(
        "--ablation",
        choices=ablation_names(),
        help="Benchmark-only query component configuration.",
    )
    parser.add_argument("--governance")
    parser.add_argument("--reuse-workspace", action="store_true")
    parser.add_argument(
        "--resume-queries",
        action="store_true",
        help="Resume matching per-query checkpoints in reusable workspaces.",
    )
    parser.add_argument("--max-download-gib", type=float, default=2.0)
    parser.add_argument("--max-extracted-gib", type=float, default=4.0)
    parser.add_argument(
        "--max-recall-regression",
        type=float,
        help="Optional allowed Recall@max-k regression versus plain BM25.",
    )
    parsed = parser.parse_args(argv)
    parsed.cutoffs = sorted(set(parsed.cutoffs or [1, 3, 5, 10, 20, 50]))
    if parsed.query_limit is not None and parsed.query_limit < 1:
        parser.error("--query-limit must be positive")
    if any(value < 1 for value in parsed.cutoffs):
        parser.error("--cutoff must be positive")
    if parsed.max_download_gib <= 0 or parsed.max_extracted_gib <= 0:
        parser.error("download and extraction budgets must be positive")
    if parsed.max_recall_regression is not None and not (
        0.0 <= parsed.max_recall_regression <= 1.0
    ):
        parser.error("--max-recall-regression must be between 0 and 1")
    if parsed.resume_queries and not parsed.reuse_workspace:
        parser.error("--resume-queries requires --reuse-workspace")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
