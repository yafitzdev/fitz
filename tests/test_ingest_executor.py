# tests/test_ingest_executor.py
"""
Tests for fitz_ai.ingest.diff.executor module.

Key tests verify that:
1. State is updated only on successful ingestion
2. Errors don't update state (per spec §10)
3. Deletions update state
4. Summary is accurate
"""

import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.diff.executor import (
    DiffIngestExecutor,
    IngestSummary,
    run_diff_ingest,
)
from fitz_ai.ingest.state import IngestStateManager


class MockVectorDBWriter:
    """Mock vector DB writer."""

    def __init__(self):
        self.upserted: List[tuple] = []
        self.deleted: List[tuple] = []

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        self.upserted.append((collection, points))

    def mark_deleted(self, collection: str, source_path: str) -> int:
        self.deleted.append((collection, source_path))
        return 1


class MockEmbedder:
    """Mock embedder."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def embed(self, text: str) -> List[float]:
        return [0.1] * self._dim


class MockParser:
    """Mock parser."""

    def __init__(self, content: str = "Test content for parsing"):
        self._content = content
        self._fail_paths: Set[str] = set()

    def fail_on(self, path: str) -> None:
        self._fail_paths.add(path)

    def parse(self, path: str) -> str:
        if path in self._fail_paths:
            raise Exception(f"Parse failed for {path}")
        return self._content


class MockChunkerPlugin:
    """Mock chunker plugin implementing ChunkerPlugin protocol."""

    plugin_name: str = "mock"

    def __init__(self, num_chunks: int = 2):
        self._num_chunks = num_chunks

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        doc_id = base_meta.get("doc_id", "unknown")
        return [
            Chunk(
                id=f"{doc_id}:{i}",
                doc_id=doc_id,
                chunk_index=i,
                content=f"Chunk {i} of {text[:20]}...",
                metadata=base_meta,
            )
            for i in range(self._num_chunks)
        ]


class TestIngestSummary:
    """Tests for IngestSummary."""

    def test_str_format(self):
        """Test string format matches spec §7.4."""
        summary = IngestSummary(
            scanned=10,
            ingested=3,
            skipped=5,
            marked_deleted=1,
            errors=1,
        )

        result = str(summary)
        assert "scanned 10" in result
        assert "ingested 3" in result
        assert "skipped 5" in result
        assert "marked_deleted 1" in result
        assert "errors 1" in result

    def test_duration(self):
        """Test duration calculation."""
        from datetime import datetime, timedelta

        summary = IngestSummary()
        summary.started_at = datetime(2024, 1, 1, 0, 0, 0)
        summary.finished_at = datetime(2024, 1, 1, 0, 0, 10)

        assert summary.duration_seconds == 10.0


class TestDiffIngestExecutor:
    """Tests for DiffIngestExecutor."""

    @pytest.fixture
    def tmp_state_manager(self, tmp_path: Path) -> IngestStateManager:
        """Create a state manager with temp path."""
        return IngestStateManager(tmp_path / "ingest.json")

    @pytest.fixture
    def executor_deps(self, tmp_state_manager):
        """Create common executor dependencies."""
        return {
            "state_manager": tmp_state_manager,
            "vector_db_writer": MockVectorDBWriter(),
            "embedder": MockEmbedder(),
            "parser": MockParser(),
            "chunker": MockChunkerPlugin(),
            "collection": "test_collection",
        }

    def test_scans_and_ingests_new_files(self, tmp_path: Path, executor_deps):
        """Test ingesting new files."""
        # Create test files
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "test2.txt").write_text("Hello")

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.scanned == 2
        assert summary.ingested == 2
        assert summary.skipped == 0
        assert summary.errors == 0

    def test_skips_existing_files(self, tmp_path: Path, executor_deps):
        """Test skipping files that exist in state with same hash."""
        # Create test file
        test_file = tmp_path / "existing.md"
        test_file.write_text("# Already indexed")

        # Pre-populate state with this file's hash
        from fitz_ai.ingest.hashing import compute_content_hash
        content_hash = compute_content_hash(test_file)

        state_manager = executor_deps["state_manager"]
        state_manager.mark_active(
            file_path=str(test_file.resolve()),
            root=str(tmp_path.resolve()),
            content_hash=content_hash,
            ext=".md",
            size_bytes=test_file.stat().st_size,
            mtime_epoch=test_file.stat().st_mtime,
        )

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.scanned == 1
        assert summary.ingested == 0
        assert summary.skipped == 1

    def test_updates_state_on_success(self, tmp_path: Path, executor_deps):
        """Test that state is updated after successful ingestion."""
        (tmp_path / "test.md").write_text("# Test")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        # Check state was saved
        state_manager = executor_deps["state_manager"]
        state_manager.load()

        active_paths = state_manager.get_active_paths(str(tmp_path.resolve()))
        assert len(active_paths) == 1

    def test_does_not_update_state_on_error(self, tmp_path: Path, executor_deps):
        """CRITICAL: State should NOT be updated for failed files (spec §10)."""
        test_file = tmp_path / "failing.md"
        test_file.write_text("# Will fail")

        # Make parser fail
        parser = MockParser()
        parser.fail_on(str(test_file.resolve()))
        executor_deps["parser"] = parser

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.errors == 1

        # State should NOT have this file
        state_manager = executor_deps["state_manager"]
        state_manager.load()
        active_paths = state_manager.get_active_paths(str(tmp_path.resolve()))
        assert len(active_paths) == 0

    def test_marks_deletions_in_vector_db_and_state(self, tmp_path: Path, executor_deps):
        """Test that deletions update both vector DB and state."""
        # Create a file and ingest it
        test_file = tmp_path / "to_delete.md"
        test_file.write_text("# Will be deleted")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        # Now delete the file
        test_file.unlink()

        # Run again
        summary = executor.run(tmp_path)

        assert summary.marked_deleted == 1

        # Check vector DB writer was called
        writer = executor_deps["vector_db_writer"]
        assert len(writer.deleted) == 1

    def test_force_mode_ingests_everything(self, tmp_path: Path, executor_deps):
        """Test force mode ingests even files in state."""
        test_file = tmp_path / "existing.md"
        test_file.write_text("# Already indexed")

        # Pre-populate state with this file
        from fitz_ai.ingest.hashing import compute_content_hash
        content_hash = compute_content_hash(test_file)

        state_manager = executor_deps["state_manager"]
        state_manager.mark_active(
            file_path=str(test_file.resolve()),
            root=str(tmp_path.resolve()),
            content_hash=content_hash,
            ext=".md",
            size_bytes=test_file.stat().st_size,
            mtime_epoch=test_file.stat().st_mtime,
        )

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path, force=True)

        assert summary.ingested == 1
        assert summary.skipped == 0

    def test_continues_on_error(self, tmp_path: Path, executor_deps):
        """Test that executor continues processing after errors."""
        (tmp_path / "good1.md").write_text("# Good 1")
        failing_file = tmp_path / "bad.md"
        failing_file.write_text("# Bad")
        (tmp_path / "good2.md").write_text("# Good 2")

        # Make parser fail on one file
        parser = MockParser()
        parser.fail_on(str(failing_file.resolve()))
        executor_deps["parser"] = parser

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        # Should have processed all files
        assert summary.scanned == 3
        assert summary.ingested == 2
        assert summary.errors == 1

    def test_upserts_to_vector_db(self, tmp_path: Path, executor_deps):
        """Test that vectors are upserted to vector DB."""
        (tmp_path / "test.md").write_text("# Test")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        writer = executor_deps["vector_db_writer"]
        assert len(writer.upserted) == 1

        collection, points = writer.upserted[0]
        assert collection == "test_collection"
        assert len(points) == 2  # MockChunker creates 2 chunks

    def test_vector_payload_contains_required_metadata(self, tmp_path: Path, executor_deps):
        """Test that vector payload contains all required metadata (spec §5.1)."""
        (tmp_path / "test.md").write_text("# Test content")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        writer = executor_deps["vector_db_writer"]
        _, points = writer.upserted[0]
        payload = points[0]["payload"]

        # Check required fields per spec §5.1
        assert "content_hash" in payload
        assert "source_path" in payload
        assert "ext" in payload
        assert "chunk_index" in payload
        assert "chunk_text_hash" in payload
        assert "parser_id" in payload
        assert "chunker_id" in payload
        assert "embedding_id" in payload
        assert "is_deleted" in payload
        assert payload["is_deleted"] is False
        assert "ingested_at" in payload

    def test_empty_directory(self, tmp_path: Path, executor_deps):
        """Test handling empty directory."""
        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.scanned == 0
        assert summary.ingested == 0


class TestRunDiffIngest:
    """Tests for run_diff_ingest convenience function."""

    def test_runs_ingestion(self, tmp_path: Path):
        """Test the convenience function."""
        (tmp_path / "test.md").write_text("# Test")

        summary = run_diff_ingest(
            source=tmp_path,
            state_manager=IngestStateManager(tmp_path / "state" / "ingest.json"),
            vector_db_writer=MockVectorDBWriter(),
            embedder=MockEmbedder(),
            parser=MockParser(),
            chunker=MockChunkerPlugin(),
            collection="test",
        )

        assert summary.scanned == 1
        assert summary.ingested == 1