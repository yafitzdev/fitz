"""Profile Fitz-Sage query latency against reusable BEIR indexes."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.beir import (
    DATASETS,
    PreparedDataset,
    load_qrels,
    load_queries,
    prepare_dataset,
)
from benchmarks.fitz_bench.timing import group_timings, summarize_timing_records
from fitz_sage.config.loader import load_engine_config
from fitz_sage.core import Query
from fitz_sage.core.paths import FitzPaths
from fitz_sage.runtime import create_engine
from fitz_sage.storage.sqlite import SqliteConnectionManager


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
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

    results = [
        profile_dataset(
            dataset,
            workspace_root=args.workspace_root,
            sample_size=args.sample_size,
            seed=args.seed,
            repeats=args.repeats,
            top_k=args.top_k,
            governance=args.governance,
            cold_probe=not args.skip_cold_probe,
        )
        for dataset in prepared
    ]
    report = {
        "schema_version": 1,
        "run": {
            "git": _git_state(),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "datasets": datasets,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "repeats": args.repeats,
            "top_k": args.top_k,
            "governance_override": args.governance,
            "cold_probe": not args.skip_cold_probe,
            "duration_seconds": time.perf_counter() - started,
        },
        "datasets": results,
    }
    _write_report(report, args.output, args.markdown)
    return 0


def profile_dataset(
    dataset: PreparedDataset,
    *,
    workspace_root: Path,
    sample_size: int,
    seed: int,
    repeats: int,
    top_k: int,
    governance: str | None,
    cold_probe: bool,
) -> dict[str, Any]:
    """Run a cold probe and deterministic warm sample on one existing index."""
    queries = load_queries(Path(dataset.source_queries))
    query_ids = list(load_qrels(Path(dataset.source_qrels)))
    selected_ids = _select_query_ids(query_ids, sample_size=sample_size, seed=seed)

    workspace = (Path(workspace_root).resolve() / dataset.name).resolve()
    collection = f"beir_{dataset.name}_v{dataset.adapter_schema_version}_source"
    manifest_path = workspace / "collections" / collection / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Reusable BEIR index not found: {manifest_path}. "
            "Run beir_benchmark with --reuse-workspace first."
        )

    _activate_workspace(workspace)
    engine = _create_engine(collection, governance=governance)
    engine.load(collection)
    cold_record: dict[str, Any] | None = None
    warm_records: list[dict[str, Any]] = []
    try:
        if cold_probe:
            query_id = selected_ids[0]
            print(f"  Cold {dataset.name}: {query_id}", flush=True)
            cold_record = _run_query(
                engine,
                query_id=query_id,
                query=queries[query_id],
                top_k=top_k,
                repeat=0,
                cold=True,
            )

        total_runs = len(selected_ids) * repeats
        completed = 0
        for repeat in range(1, repeats + 1):
            for query_id in selected_ids:
                completed += 1
                record = _run_query(
                    engine,
                    query_id=query_id,
                    query=queries[query_id],
                    top_k=top_k,
                    repeat=repeat,
                    cold=False,
                )
                warm_records.append(record)
                print(
                    f"  Warm {dataset.name}: {completed}/{total_runs} "
                    f"({record['total_seconds']:.2f}s)",
                    flush=True,
                )
    finally:
        engine.stop_background_enrichment()

    return {
        "dataset": dataset.as_dict(),
        "workspace": str(workspace),
        "collection": collection,
        "selection": {
            "available_judged_queries": len(query_ids),
            "sampled_queries": len(selected_ids),
            "query_ids": selected_ids,
            "seed": seed,
            "repeats": repeats,
            "warm_runs": len(warm_records),
        },
        "cold_probe": cold_record,
        "warm_summary": summarize_timing_records(warm_records),
        "warm_records": warm_records,
    }


def _run_query(
    engine: Any,
    *,
    query_id: str,
    query: str,
    top_k: int,
    repeat: int,
    cold: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    pack = engine.evidence(Query(text=query), top_k=top_k)
    total = time.perf_counter() - started
    raw_timings = {str(name): float(duration) for name, duration in pack.timings.items()}
    grouped, overlap = group_timings(raw_timings, total_seconds=total)
    return {
        "query_id": query_id,
        "query": query,
        "query_characters": len(query),
        "repeat": repeat,
        "cold": cold,
        "total_seconds": total,
        "stage_seconds": raw_timings,
        "grouped_seconds": grouped,
        "timing_overlap_seconds": overlap,
    }


def _select_query_ids(
    query_ids: list[str],
    *,
    sample_size: int,
    seed: int,
) -> list[str]:
    """Select a stable random sample while preserving dataset order."""
    if sample_size >= len(query_ids):
        return list(query_ids)
    generator = random.Random(seed)
    selected = set(generator.sample(range(len(query_ids)), sample_size))
    return [query_id for index, query_id in enumerate(query_ids) if index in selected]


def _activate_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    SqliteConnectionManager.reset_instance()
    FitzPaths.set_workspace(workspace)


def _create_engine(collection: str, *, governance: str | None) -> Any:
    config = load_engine_config("fitz_krag")
    values = config.model_dump()
    values["collection"] = collection
    if governance is not None:
        values["governance"] = governance
    return create_engine("fitz_krag", config=type(config)(**values))


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


def _write_report(report: dict[str, Any], output: Path, markdown: Path | None) -> None:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown = Path(markdown).resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "markdown": str(markdown) if markdown else None}))


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BEIR Query Timing",
        "",
        "Warm timings exclude the explicit first-query cold probe.",
        "",
        "| Dataset | Runs | Total p50 | Total p95 | Qwen mean | Rerank mean | Pyrrho mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in report["datasets"]:
        summary = result["warm_summary"]
        total = summary.get("total_latency", {})
        groups = summary.get("stage_groups", {})
        pyrrho_mean = groups.get("pyrrho_planning", {}).get("mean_seconds", 0.0) + groups.get(
            "pyrrho_decision", {}
        ).get("mean_seconds", 0.0)
        lines.append(
            "| {dataset} | {runs} | {p50:.2f}s | {p95:.2f}s | {qwen:.2f}s | "
            "{rerank:.2f}s | {pyrrho:.2f}s |".format(
                dataset=result["dataset"]["name"],
                runs=summary.get("runs", 0),
                p50=total.get("p50_seconds", 0.0),
                p95=total.get("p95_seconds", 0.0),
                qwen=groups.get("semantic_expansion", {}).get("mean_seconds", 0.0),
                rerank=groups.get("rerank", {}).get("mean_seconds", 0.0),
                pyrrho=pyrrho_mean,
            )
        )

    lines.extend(
        [
            "",
            "| Dataset | Cold total | Cold Qwen | Cold rerank | Cold Pyrrho |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in report["datasets"]:
        cold = result.get("cold_probe") or {}
        groups = cold.get("grouped_seconds", {})
        lines.append(
            "| {dataset} | {total:.2f}s | {qwen:.2f}s | {rerank:.2f}s | {pyrrho:.2f}s |".format(
                dataset=result["dataset"]["name"],
                total=float(cold.get("total_seconds", 0.0)),
                qwen=float(groups.get("semantic_expansion", 0.0)),
                rerank=float(groups.get("rerank", 0.0)),
                pyrrho=float(groups.get("pyrrho_planning", 0.0))
                + float(groups.get("pyrrho_decision", 0.0)),
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to profile. Repeat; defaults to all supported datasets.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".benchmark-data/beir"))
    parser.add_argument("--workspace-root", type=Path, default=Path(".bench_workspace/beir"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/beir_timing_latest.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("benchmarks/results/beir_timing_latest.md"),
    )
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--governance")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--skip-cold-probe", action="store_true")
    parser.add_argument("--max-download-gib", type=float, default=2.0)
    parser.add_argument("--max-extracted-gib", type=float, default=4.0)
    parsed = parser.parse_args(argv)
    if parsed.sample_size < 1 or parsed.repeats < 1 or parsed.top_k < 1:
        parser.error("sample size, repeats, and top-k must be positive")
    if parsed.max_download_gib <= 0 or parsed.max_extracted_gib <= 0:
        parser.error("download and extraction budgets must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
