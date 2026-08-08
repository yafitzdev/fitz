"""Verify that source indexing converges after an abrupt process exit."""

from __future__ import annotations

import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.ingestion_benchmark import (
    manifest_inventory,
    sqlite_counts,
)
from benchmarks.fitz_bench.ingestion_worker import CRASH_EXIT_CODE
from fitz_sage.core.paths import FitzPaths
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.runtime import create_engine
from fitz_sage.storage.sqlite import SqliteConnectionManager


def run_recovery_benchmark(
    source: Path,
    *,
    workspace: Path,
    parser: str = "cpu",
    crash_after_indexed: int = 10,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Crash after durable file writes, resume, and compare with a clean run."""
    if crash_after_indexed < 1:
        raise ValueError("crash_after_indexed must be at least 1")

    source = Path(source).resolve()
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    collection = f"recovery_bench_{uuid.uuid4().hex[:8]}"
    worker_report = workspace / "unexpected-worker-completion.json"
    worker_log = workspace / "crash-worker.log"
    worker_report.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-m",
        "benchmarks.fitz_bench.ingestion_worker",
        "--source",
        str(source),
        "--workspace",
        str(workspace),
        "--collection",
        collection,
        "--parser",
        parser,
        "--crash-after-indexed",
        str(crash_after_indexed),
        "--output",
        str(worker_report),
    ]

    with worker_log.open("wb") as log:
        crashed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    partial_counts = sqlite_counts(workspace, collection)
    crash_exercised = (
        crashed.returncode == CRASH_EXIT_CODE
        and 0 < partial_counts["raw_files"]
        and not worker_report.exists()
    )

    FitzPaths.set_workspace(workspace)
    SqliteConnectionManager.reset_instance()
    try:
        engine = create_engine(
            "fitz_krag",
            config=FitzKragConfig(collection=collection, parser=parser),
        )
        engine.load(collection)
        resume_started = time.perf_counter()
        manifest = engine.point(
            source,
            collection=collection,
            start_worker=False,
        )
        resume_seconds = time.perf_counter() - resume_started
        status = dict(engine.indexing_status())
        inventory = manifest_inventory(manifest)
        recovered_counts = sqlite_counts(workspace, collection)
        engine.stop_background_enrichment()
    finally:
        SqliteConnectionManager.reset_instance()
        FitzPaths.reset()

    actual_signature = {
        "indexed_files": int(status.get("indexed", 0)),
        "failed_files": int(status.get("failed", 0)),
        "unsupported_files": int(status.get("unsupported", 0)),
        "indexed_bytes": int(inventory["indexed_bytes"]),
        "by_extension": inventory["by_extension"],
        "sqlite_counts": recovered_counts,
    }
    expected_matches = expected is None or actual_signature == expected
    no_orphan_raw_files = recovered_counts["raw_files"] == int(status.get("indexed", 0))
    passed = (
        crash_exercised
        and bool(status.get("query_ready"))
        and int(status.get("pending", 0)) == 0
        and no_orphan_raw_files
        and expected_matches
    )
    return {
        "source": str(source),
        "workspace": str(workspace),
        "collection": collection,
        "parser": parser,
        "crash_after_indexed": crash_after_indexed,
        "crash_exit_code": crashed.returncode,
        "expected_crash_exit_code": CRASH_EXIT_CODE,
        "crash_exercised": crash_exercised,
        "partial_sqlite_counts": partial_counts,
        "resume_seconds": resume_seconds,
        "status": status,
        "actual": actual_signature,
        "expected_matches": expected_matches,
        "no_orphan_raw_files": no_orphan_raw_files,
        "worker_log": str(worker_log),
        "passed": passed,
    }
