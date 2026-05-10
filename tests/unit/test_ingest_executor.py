# tests/unit/test_ingest_executor.py
"""
Tests for fitz_sage.ingestion.diff.executor module.

Key tests verify that:
1. State is updated only on successful ingestion
2. Errors don't update state (per spec §10)
3. Deletions update state
4. Summary is accurate
5. ChunkingRouter is used for file-type routing
"""

from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

from tests.conftest import POSTGRES_DEPS_AVAILABLE, SKIP_POSTGRES_REASON

# Skip entire module if postgres dependencies not available
# (IngestStateManager and DiffIngestExecutor use PostgreSQL storage)
if not POSTGRES_DEPS_AVAILABLE:
    pytest.skip(SKIP_POSTGRES_REASON, allow_module_level=True)

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.ingestion.chunking.plugins.default.simple import SimpleChunker
from fitz_sage.ingestion.chunking.router import ChunkingRouter
from fitz_sage.ingestion.diff.executor import (
    DiffIngestExecutor,
    IngestSummary,
    run_diff_ingest,
)
from fitz_sage.ingestion.source.base import SourceFile
from fitz_sage.ingestion.state import IngestStateManager


class MockVectorDBWriter:
    """Mock vector DB writer."""

    def __init__(self):
        self.upserted: List[tuple] = []
        self.deleted: List[tuple] = []

    def upsert(
        self, collection: str, points: List[Dict[str, Any]], defer_persist: bool = False
    ) -> None:
        self.upserted.append((collection, points))

    def mark_deleted(self, collection: str, source_path: str) -> int:
        self.deleted.append((collection, source_path))
        return 1


class MockEmbedder:
    """Mock embedder."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def embed(self, text: str, **kwargs) -> List[float]:
        return [0.1] * self._dim

    def embed_batch(self, texts: List[str], **kwargs) -> List[List[float]]:
        return [[0.1] * self._dim for _ in texts]


class MockParserRouter:
    """Mock parser router for testing."""

    def __init__(self, content: str = "Test content for parsing"):
        self._content = content
        self._fail_paths: Set[str] = set()

    def fail_on(self, path: str) -> None:
        self._fail_paths.add(path)

    def parse(self, source_file: SourceFile) -> ParsedDocument:
        path_str = str(source_file.local_path)
        if path_str in self._fail_paths:
            raise Exception(f"Parse failed for {path_str}")
        return ParsedDocument(
            source=source_file.uri,
            elements=[DocumentElement(type=ElementType.TEXT, content=self._content)],
            metadata={"source_file": path_str},
        )

    def get_parser_id(self, ext: str) -> str:
        return f"mock:{ext[1:] if ext.startswith('.') else ext}:v1"


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
        from datetime import datetime, timedelta, timezone

        summary = IngestSummary()
        summary.started_at = datetime.now(timezone.utc)
        summary.finished_at = summary.started_at + timedelta(seconds=5)

        assert summary.duration_seconds == pytest.approx(5.0, abs=0.1)


class TestDiffIngestExecutor:
    """Tests for DiffIngestExecutor."""

    @pytest.fixture
    def executor_deps(self, tmp_path: Path):
        """Common dependencies for executor tests."""
        state_path = tmp_path / "state" / "ingest.json"
        state_manager = IngestStateManager(state_path)
        state_manager.load()

        # Create a router with recursive chunker (better than simple)
        from fitz_sage.ingestion.chunking.plugins.default.recursive import RecursiveChunker

        router = ChunkingRouter(
            chunker_map={},
            default_chunker=RecursiveChunker(chunk_size=1000, chunk_overlap=200),
            warn_on_fallback=False,
        )

        return {
            "state_manager": state_manager,
            "vector_db_writer": MockVectorDBWriter(),
            "embedder": MockEmbedder(),
            "parser_router": MockParserRouter(),
            "chunking_router": router,
            "collection": "test_collection",
            "embedding_id": "test:embedding",
            "vector_db_id": "test:vectordb",
        }

    def test_scans_and_ingests_new_files(self, tmp_path: Path, executor_deps):
        """Test ingesting new files."""
        (tmp_path / "test.md").write_text("# Test")
        (tmp_path / "test2.txt").write_text("Hello")

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.scanned >= 1
        assert summary.ingested >= 1
        assert summary.errors == 0

    def test_skips_existing_files(self, tmp_path: Path, executor_deps):
        """Test skipping files that exist in state with same hash."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Already indexed content")

        from fitz_sage.ingestion.hashing import compute_content_hash

        content_hash = compute_content_hash(test_file)

        state_manager = executor_deps["state_manager"]
        state_manager.mark_active(
            file_path=str(test_file.resolve()),
            root=str(tmp_path.resolve()),
            content_hash=content_hash,
            ext=".txt",
            size_bytes=test_file.stat().st_size,
            mtime_epoch=test_file.stat().st_mtime,
            chunker_id="recursive:1000:200",
            parser_id="mock:txt:v1",
            embedding_id="test:embedding",
            vector_db_id="test:vectordb",
            collection="test_collection",
        )

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.skipped == 1
        assert summary.ingested == 0

    def test_updates_state_on_success(self, tmp_path: Path, executor_deps):
        """Test that state is updated after successful ingestion."""
        (tmp_path / "test.md").write_text("# Test")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        state_manager = executor_deps["state_manager"]
        active_paths = state_manager.get_active_paths(str(tmp_path.resolve()))

        assert len(active_paths) >= 1

    def test_does_not_update_state_on_error(self, tmp_path: Path, executor_deps):
        """CRITICAL: State should NOT be updated for failed files (spec §10)."""
        test_file = tmp_path / "failing.md"
        test_file.write_text("# Will fail")

        parser_router = MockParserRouter()
        parser_router.fail_on(str(test_file.resolve()))
        executor_deps["parser_router"] = parser_router

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.errors == 1

        state_manager = executor_deps["state_manager"]
        entry = state_manager.get_file_entry(str(tmp_path.resolve()), str(test_file.resolve()))
        assert entry is None

    def test_marks_deletions_in_state(self, tmp_path: Path, executor_deps):
        """Test that deletions update state."""
        test_file = tmp_path / "to_delete.md"
        test_file.write_text("# Will be deleted")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        test_file.unlink()

        executor.run(tmp_path)

        state_manager = executor_deps["state_manager"]
        entry = state_manager.get_file_entry(str(tmp_path.resolve()), str(test_file.resolve()))
        assert entry is not None
        assert entry.status.value == "deleted"

    def test_force_mode_ingests_everything(self, tmp_path: Path, executor_deps):
        """Test force mode ingests even files in state."""
        test_file = tmp_path / "existing.txt"
        test_file.write_text("Already indexed content")

        from fitz_sage.ingestion.hashing import compute_content_hash

        content_hash = compute_content_hash(test_file)

        state_manager = executor_deps["state_manager"]
        state_manager.mark_active(
            file_path=str(test_file.resolve()),
            root=str(tmp_path.resolve()),
            content_hash=content_hash,
            ext=".txt",
            size_bytes=test_file.stat().st_size,
            mtime_epoch=test_file.stat().st_mtime,
            chunker_id="recursive:1000:200",
            parser_id="mock:txt:v1",
            embedding_id="test:embedding",
            vector_db_id="test:vectordb",
            collection="test_collection",
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

        parser_router = MockParserRouter()
        parser_router.fail_on(str(failing_file.resolve()))
        executor_deps["parser_router"] = parser_router

        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.ingested == 2
        assert summary.errors == 1

    def test_upserts_to_vector_db(self, tmp_path: Path, executor_deps):
        """Test that vectors are upserted to vector DB."""
        (tmp_path / "test.md").write_text("# Test")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        writer = executor_deps["vector_db_writer"]
        assert len(writer.upserted) > 0

    def test_vector_payload_contains_required_metadata(self, tmp_path: Path, executor_deps):
        """Test that vector payload contains all required metadata (spec §5.1)."""
        (tmp_path / "test.md").write_text("# Test content")

        executor = DiffIngestExecutor(**executor_deps)
        executor.run(tmp_path)

        writer = executor_deps["vector_db_writer"]
        assert len(writer.upserted) > 0

        _, points = writer.upserted[0]
        payload = points[0]["payload"]

        assert "content" in payload
        assert "doc_id" in payload
        assert "chunk_index" in payload
        assert "content_hash" in payload
        assert "source_path" in payload
        assert "chunker_id" in payload
        assert "parser_id" in payload
        assert "embedding_id" in payload

    def test_empty_directory(self, tmp_path: Path, executor_deps):
        """Test handling empty directory."""
        executor = DiffIngestExecutor(**executor_deps)
        summary = executor.run(tmp_path)

        assert summary.scanned == 0
        assert summary.ingested == 0
        assert summary.errors == 0


class TestRunDiffIngest:
    """Tests for run_diff_ingest convenience function."""

    def test_runs_ingestion(self, tmp_path: Path):
        """Test the convenience function."""
        (tmp_path / "test.md").write_text("# Test")

        router = ChunkingRouter(
            chunker_map={},
            default_chunker=SimpleChunker(chunk_size=1000),
            warn_on_fallback=False,
        )

        state_manager = IngestStateManager(tmp_path / "state" / "ingest.json")
        state_manager.load()

        summary = run_diff_ingest(
            source=tmp_path,
            state_manager=state_manager,
            vector_db_writer=MockVectorDBWriter(),
            embedder=MockEmbedder(),
            parser_router=MockParserRouter(),
            chunking_router=router,
            collection="test",
            embedding_id="test:embedding",
        )

        assert summary.ingested >= 1
        assert summary.errors == 0
