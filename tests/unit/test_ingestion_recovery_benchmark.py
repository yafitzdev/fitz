"""Tests for hard-crash source-index recovery."""

from __future__ import annotations

from benchmarks.fitz_bench.ingestion_benchmark import run_benchmark
from benchmarks.fitz_bench.recovery_benchmark import run_recovery_benchmark


def test_indexing_converges_after_hard_process_exit(tmp_path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    for index in range(6):
        (source / f"document-{index}.md").write_text(
            f"# Document {index}\n\nDurable content {index}.",
            encoding="utf-8",
        )

    baseline = run_benchmark(
        source,
        workspace=tmp_path / "baseline",
        iterations=1,
        target_files_per_second=0,
    )
    run = baseline["iterations"][0]
    expected = {
        "indexed_files": run["indexed_files"],
        "failed_files": run["failed_files"],
        "unsupported_files": run["unsupported_files"],
        "indexed_bytes": run["indexed_bytes"],
        "by_extension": run["by_extension"],
        "sqlite_counts": run["sqlite_counts"],
    }

    recovery_workspace = tmp_path / "recovery"
    recovery_workspace.mkdir()
    (recovery_workspace / "unexpected-worker-completion.json").write_text(
        "stale",
        encoding="utf-8",
    )
    recovery = run_recovery_benchmark(
        source,
        workspace=recovery_workspace,
        crash_after_indexed=2,
        expected=expected,
    )

    assert recovery["crash_exercised"] is True
    assert recovery["partial_sqlite_counts"]["raw_files"] == 2
    assert recovery["expected_matches"] is True
    assert recovery["no_orphan_raw_files"] is True
    assert recovery["passed"] is True
