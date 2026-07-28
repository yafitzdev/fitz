"""Tests for the query-ready ingestion benchmark report."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from benchmarks.fitz_bench.ingestion_benchmark import run_benchmark


def test_ingestion_benchmark_measures_point_without_enrichment(tmp_path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "one.md").write_text("# One\nBody", encoding="utf-8")

    engine = MagicMock()
    engine.indexing_status.return_value = {
        "indexed": 1,
        "failed": 0,
        "unsupported": 0,
        "query_ready": True,
    }

    with patch(
        "benchmarks.fitz_bench.ingestion_benchmark.create_engine",
        return_value=engine,
    ):
        report = run_benchmark(
            source,
            workspace=tmp_path / ".fitz",
            iterations=1,
            target_files_per_second=0,
        )

    assert report["gate"]["passed"] is True
    assert engine.point.call_count == 2
    assert all(call.kwargs["start_worker"] is False for call in engine.point.call_args_list)
    engine.continue_enrichment.assert_not_called()
    engine.stop_background_enrichment.assert_called_once()
