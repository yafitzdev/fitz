"""Measure cold query-ready indexing independently from background enrichment."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fitz_sage.core.paths import FitzPaths
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.runtime import create_engine
from fitz_sage.storage.sqlite import SqliteConnectionManager


def main(argv: list[str] | None = None) -> int:
    """Run repeated cold source-index builds and print a JSON report."""
    args = _parse_args(argv)
    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")

    with tempfile.TemporaryDirectory(prefix="fitz-ingestion-bench-") as temp_dir:
        workspace = args.workspace.resolve() if args.workspace else Path(temp_dir) / ".fitz"
        report = run_benchmark(
            source,
            workspace=workspace,
            iterations=args.iterations,
            parser=args.parser,
            target_files_per_second=args.target_files_per_second,
        )

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["gate"]["passed"] else 1


def run_benchmark(
    source: Path,
    *,
    workspace: Path,
    iterations: int = 3,
    parser: str = "cpu",
    target_files_per_second: float = 1.0,
) -> dict[str, Any]:
    """Return cold-index and no-change re-point measurements."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    source = Path(source).resolve()
    workspace = Path(workspace).resolve()
    FitzPaths.set_workspace(workspace)
    SqliteConnectionManager.reset_instance()
    runs: list[dict[str, Any]] = []
    try:
        for iteration in range(1, iterations + 1):
            collection = f"ingestion_bench_{iteration}_{uuid.uuid4().hex[:8]}"
            setup_started = time.perf_counter()
            engine = create_engine(
                "fitz_krag",
                config=FitzKragConfig(collection=collection, parser=parser),
            )
            engine.load(collection)
            setup_seconds = time.perf_counter() - setup_started

            point_started = time.perf_counter()
            manifest = engine.point(
                source,
                collection=collection,
                start_worker=False,
            )
            point_seconds = time.perf_counter() - point_started
            status = dict(engine.indexing_status())
            indexed_bytes = _indexed_bytes(manifest)

            repoint_started = time.perf_counter()
            engine.point(
                source,
                collection=collection,
                start_worker=False,
            )
            repoint_seconds = time.perf_counter() - repoint_started

            indexed = int(status.get("indexed", 0))
            runs.append(
                {
                    "iteration": iteration,
                    "setup_seconds": setup_seconds,
                    "query_ready_seconds": point_seconds,
                    "no_change_repoint_seconds": repoint_seconds,
                    "indexed_files": indexed,
                    "indexed_bytes": indexed_bytes,
                    "failed_files": int(status.get("failed", 0)),
                    "unsupported_files": int(status.get("unsupported", 0)),
                    "query_ready": bool(status.get("query_ready")),
                    "files_per_second": indexed / point_seconds if point_seconds > 0 else None,
                    "mib_per_second": (
                        indexed_bytes / (1024 * 1024) / point_seconds
                        if point_seconds > 0
                        else None
                    ),
                }
            )
            engine.stop_background_enrichment()

        rates = [
            float(run["files_per_second"])
            for run in runs
            if run["files_per_second"] is not None
        ]
        byte_rates = [
            float(run["mib_per_second"])
            for run in runs
            if run["mib_per_second"] is not None
        ]
        durations = [float(run["query_ready_seconds"]) for run in runs]
        failures = sum(int(run["failed_files"]) for run in runs)
        all_query_ready = all(bool(run["query_ready"]) for run in runs)
        median_rate = statistics.median(rates) if rates else 0.0
        return {
            "source": str(source),
            "workspace": str(workspace),
            "parser": parser,
            "iterations": runs,
            "summary": {
                "median_query_ready_seconds": statistics.median(durations),
                "median_files_per_second": median_rate,
                "min_files_per_second": min(rates) if rates else 0.0,
                "max_files_per_second": max(rates) if rates else 0.0,
                "median_mib_per_second": (
                    statistics.median(byte_rates) if byte_rates else 0.0
                ),
                "total_indexing_failures": failures,
            },
            "gate": {
                "target_files_per_second": target_files_per_second,
                "passed": (
                    all_query_ready
                    and failures == 0
                    and median_rate >= target_files_per_second
                ),
            },
        }
    finally:
        SqliteConnectionManager.reset_instance()
        FitzPaths.reset()


def _indexed_bytes(manifest: Any) -> int:
    """Return indexed source bytes from a manifest-like result."""
    entries_method = getattr(manifest, "entries", None)
    if not callable(entries_method):
        return 0
    entries = entries_method()
    if not isinstance(entries, dict):
        return 0
    return sum(
        int(getattr(entry, "size_bytes", 0) or 0)
        for entry in entries.values()
        if getattr(getattr(entry, "state", None), "value", None) == "indexed"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("benchmarks/corpora/core"),
        help="File or directory to index.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Persist benchmark databases here instead of using a temporary workspace.",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--parser",
        choices=("cpu", "docling", "docling_vision", "glm_ocr"),
        default="cpu",
    )
    parser.add_argument("--target-files-per-second", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
