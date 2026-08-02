"""Measure cold query-ready indexing independently from background enrichment."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sqlite3
import statistics
import sys
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
            max_failure_rate=args.max_failure_rate,
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
    max_failure_rate: float = 0.0,
) -> dict[str, Any]:
    """Return cold-index and no-change re-point measurements."""
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not 0 <= max_failure_rate <= 1:
        raise ValueError("max_failure_rate must be between 0 and 1")

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
            inventory = manifest_inventory(manifest)
            indexed_bytes = int(inventory["indexed_bytes"])
            database_bytes = _database_bytes(workspace, collection)
            record_counts = sqlite_counts(workspace, collection)

            repoint_started = time.perf_counter()
            repoint_manifest = engine.point(
                source,
                collection=collection,
                start_worker=False,
            )
            repoint_seconds = time.perf_counter() - repoint_started
            repoint_status = dict(engine.indexing_status())
            repoint_inventory = manifest_inventory(repoint_manifest)
            repoint_counts = sqlite_counts(workspace, collection)
            repoint_unchanged = _index_signature(
                status, inventory, record_counts
            ) == _index_signature(repoint_status, repoint_inventory, repoint_counts)

            indexed = int(status.get("indexed", 0))
            runs.append(
                {
                    "iteration": iteration,
                    "collection": collection,
                    "setup_seconds": setup_seconds,
                    "query_ready_seconds": point_seconds,
                    "no_change_repoint_seconds": repoint_seconds,
                    "indexed_files": indexed,
                    "indexed_bytes": indexed_bytes,
                    "failed_files": int(status.get("failed", 0)),
                    "failure_details": list(status.get("failed_files", [])),
                    "unsupported_files": int(status.get("unsupported", 0)),
                    "unsupported_details": list(status.get("unsupported_files", [])),
                    "query_ready": bool(status.get("query_ready")),
                    "files_per_second": indexed / point_seconds if point_seconds > 0 else None,
                    "mib_per_second": (
                        indexed_bytes / (1024 * 1024) / point_seconds if point_seconds > 0 else None
                    ),
                    "database_bytes": database_bytes,
                    "database_to_source_ratio": (
                        database_bytes / indexed_bytes if indexed_bytes else None
                    ),
                    "process_peak_rss_bytes": process_peak_rss_bytes(),
                    "file_size_bytes": inventory["file_size_bytes"],
                    "by_extension": inventory["by_extension"],
                    "sqlite_counts": record_counts,
                    "repoint_unchanged": repoint_unchanged,
                    "repoint_sqlite_counts": repoint_counts,
                }
            )
            engine.stop_background_enrichment()

        rates = [
            float(run["files_per_second"]) for run in runs if run["files_per_second"] is not None
        ]
        byte_rates = [
            float(run["mib_per_second"]) for run in runs if run["mib_per_second"] is not None
        ]
        durations = [float(run["query_ready_seconds"]) for run in runs]
        failures = sum(int(run["failed_files"]) for run in runs)
        attempted = sum(int(run["indexed_files"]) + int(run["failed_files"]) for run in runs)
        failure_rate = failures / attempted if attempted else 0.0
        all_query_ready = all(bool(run["query_ready"]) for run in runs)
        all_repoints_unchanged = all(bool(run["repoint_unchanged"]) for run in runs)
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
                "median_mib_per_second": (statistics.median(byte_rates) if byte_rates else 0.0),
                "max_process_peak_rss_bytes": max(
                    (int(run["process_peak_rss_bytes"]) for run in runs),
                    default=0,
                ),
                "total_indexing_failures": failures,
                "indexing_failure_rate": failure_rate,
                "all_repoints_unchanged": all_repoints_unchanged,
            },
            "gate": {
                "target_files_per_second": target_files_per_second,
                "max_failure_rate": max_failure_rate,
                "passed": (
                    all_query_ready
                    and all_repoints_unchanged
                    and failure_rate <= max_failure_rate
                    and median_rate >= target_files_per_second
                ),
            },
        }
    finally:
        SqliteConnectionManager.reset_instance()
        FitzPaths.reset()


def manifest_inventory(manifest: Any) -> dict[str, Any]:
    """Return state, extension, and size metrics from a manifest-like result."""
    entries_method = getattr(manifest, "entries", None)
    if not callable(entries_method):
        return _empty_inventory()
    entries = entries_method()
    if not isinstance(entries, dict):
        return _empty_inventory()

    by_extension: dict[str, dict[str, int]] = {}
    indexed_sizes: list[int] = []
    for entry in entries.values():
        state = str(getattr(getattr(entry, "state", None), "value", "unknown"))
        extension = str(getattr(entry, "file_type", "") or "(none)").lower()
        size_bytes = int(getattr(entry, "size_bytes", 0) or 0)
        extension_stats = by_extension.setdefault(
            extension,
            {
                "discovered": 0,
                "indexed": 0,
                "failed": 0,
                "unsupported": 0,
                "bytes": 0,
            },
        )
        extension_stats["discovered"] += 1
        if state in extension_stats:
            extension_stats[state] += 1
        if state == "indexed":
            extension_stats["bytes"] += size_bytes
            indexed_sizes.append(size_bytes)

    indexed_sizes.sort()
    return {
        "indexed_bytes": sum(indexed_sizes),
        "file_size_bytes": {
            "minimum": indexed_sizes[0] if indexed_sizes else 0,
            "median": statistics.median(indexed_sizes) if indexed_sizes else 0,
            "p95": _nearest_rank(indexed_sizes, 0.95),
            "maximum": indexed_sizes[-1] if indexed_sizes else 0,
        },
        "by_extension": dict(sorted(by_extension.items())),
    }


def sqlite_counts(workspace: Path, collection: str) -> dict[str, int]:
    """Count durable retrieval units in one benchmark collection database."""
    database = Path(workspace) / "sqlite" / f"fitz_{collection}.db"
    tables = {
        "raw_files": "krag_raw_files",
        "symbols": "krag_symbol_index",
        "sections": "krag_section_index",
        "tables": "krag_table_index",
    }
    if not database.exists():
        return {key: 0 for key in tables}

    counts: dict[str, int] = {}
    try:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for key, table in tables.items():
                try:
                    row = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                    counts[key] = int(row[0]) if row else 0
                except sqlite3.DatabaseError:
                    counts[key] = 0
    except sqlite3.DatabaseError:
        return {key: 0 for key in tables}
    return counts


def _index_signature(
    status: dict[str, Any],
    inventory: dict[str, Any],
    record_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "indexed_files": int(status.get("indexed", 0)),
        "failed_files": int(status.get("failed", 0)),
        "unsupported_files": int(status.get("unsupported", 0)),
        "indexed_bytes": int(inventory["indexed_bytes"]),
        "by_extension": inventory["by_extension"],
        "sqlite_counts": record_counts,
    }


def process_peak_rss_bytes() -> int:
    """Return peak resident memory for this process when the OS exposes it."""
    if os.name == "nt":
        return _windows_peak_rss_bytes()
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    except (ImportError, OSError, ValueError):
        return 0


def _windows_peak_rss_bytes() -> int:
    """Read PeakWorkingSetSize through the Windows process API."""
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        current_process = kernel32.GetCurrentProcess()
        succeeded = psapi.GetProcessMemoryInfo(
            current_process,
            ctypes.byref(counters),
            counters.cb,
        )
    except (AttributeError, OSError):
        return 0
    return int(counters.PeakWorkingSetSize) if succeeded else 0


def _database_bytes(workspace: Path, collection: str) -> int:
    database = Path(workspace) / "sqlite" / f"fitz_{collection}.db"
    return sum(
        path.stat().st_size for path in database.parent.glob(f"{database.name}*") if path.is_file()
    )


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    index = max(0, min(len(values) - 1, math.ceil(percentile * len(values)) - 1))
    return int(values[index])


def _empty_inventory() -> dict[str, Any]:
    return {
        "indexed_bytes": 0,
        "file_size_bytes": {
            "minimum": 0,
            "median": 0,
            "p95": 0,
            "maximum": 0,
        },
        "by_extension": {},
    }


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
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.0,
        help="Maximum supported-file indexing failure rate accepted by the gate.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
