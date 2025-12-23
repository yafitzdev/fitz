# tests/test_ingest_differ.py
"""
Tests for fitz_ai.ingest.diff.differ module.

Key tests verify that:
1. State file is used for skip decisions (authoritative source)
2. Differ only computes actions, doesn't execute
3. Deletion detection works via state comparison
"""

import pytest
from typing import Optional, Set

from fitz_ai.ingest.diff.differ import (
    Differ,
    DiffResult,
    FileCandidate,
    StateReader,
    compute_diff,
)
from fitz_ai.ingest.diff.scanner import ScannedFile


class MockFileEntry:
    """Mock file entry for testing."""

    def __init__(self, content_hash: str):
        self.content_hash = content_hash


class MockStateReader:
    """Mock state reader for testing."""

    def __init__(
            self,
            active_paths: Set[str] | None = None,
            file_entries: dict[str, MockFileEntry] | None = None,
            parser_id: str = "md.v1",
            chunker_id: str = "tokens_800_120",
            embedding_id: str = "openai:text-embedding-3-small",
    ):
        self._active_paths = active_paths or set()
        self._file_entries = file_entries or {}
        self._parser_id = parser_id
        self._chunker_id = chunker_id
        self._embedding_id = embedding_id

    def get_active_paths(self, root: str) -> Set[str]:
        return self._active_paths

    def get_file_entry(self, root: str, file_path: str) -> Optional[MockFileEntry]:
        return self._file_entries.get(file_path)

    def get_parser_id(self, ext: str) -> str:
        return self._parser_id

    def get_chunker_id(self, ext: str) -> str:
        return self._chunker_id

    def get_embedding_id(self) -> str:
        return self._embedding_id


def make_scanned_file(
        path: str,
        root: str = "/root",
        ext: str = ".md",
        content_hash: str = "sha256:abc123",
) -> ScannedFile:
    """Helper to create a ScannedFile."""
    return ScannedFile(
        path=path,
        root=root,
        ext=ext,
        size_bytes=100,
        mtime_epoch=1234567890.0,
        content_hash=content_hash,
    )


class TestFileCandidate:
    """Tests for FileCandidate dataclass."""

    def test_from_scanned(self):
        """Test creating FileCandidate from ScannedFile."""
        scanned = make_scanned_file("/root/test.md")

        candidate = FileCandidate.from_scanned(
            scanned,
            parser_id="md.v1",
            chunker_id="tokens_800_120",
            embedding_id="openai:text-embedding-3-small",
        )

        assert candidate.path == "/root/test.md"
        assert candidate.content_hash == "sha256:abc123"
        assert candidate.parser_id == "md.v1"
        assert candidate.chunker_id == "tokens_800_120"
        assert candidate.embedding_id == "openai:text-embedding-3-small"


class TestDiffResult:
    """Tests for DiffResult dataclass."""

    def test_summary(self):
        """Test summary property."""
        result = DiffResult(
            to_ingest=[FileCandidate.from_scanned(
                make_scanned_file("/root/a.md"),
                "md.v1", "tokens_800_120", "openai:text-embedding-3-small"
            )],
            to_skip=[FileCandidate.from_scanned(
                make_scanned_file("/root/b.md"),
                "md.v1", "tokens_800_120", "openai:text-embedding-3-small"
            )],
            to_mark_deleted=["/root/c.md", "/root/d.md"],
        )

        assert result.summary == "ingest=1, skip=1, delete=2"


class TestDiffer:
    """Tests for Differ class."""

    def test_skips_file_with_matching_hash_in_state(self):
        """File should be skipped if state has entry with same hash."""
        state = MockStateReader(
            active_paths={"/root/existing.md"},
            file_entries={
                "/root/existing.md": MockFileEntry("sha256:exists"),
            },
        )

        differ = Differ(state)

        scanned = [
            make_scanned_file("/root/existing.md", content_hash="sha256:exists"),
        ]

        result = differ.compute_diff(scanned)

        # Should be skipped (same hash in state)
        assert len(result.to_skip) == 1
        assert len(result.to_ingest) == 0

    def test_ingests_file_with_different_hash(self):
        """File should be ingested if state has entry with different hash."""
        state = MockStateReader(
            active_paths={"/root/changed.md"},
            file_entries={
                "/root/changed.md": MockFileEntry("sha256:old_hash"),
            },
        )

        differ = Differ(state)

        scanned = [
            make_scanned_file("/root/changed.md", content_hash="sha256:new_hash"),
        ]

        result = differ.compute_diff(scanned)

        # Should be ingested (hash changed)
        assert len(result.to_ingest) == 1
        assert len(result.to_skip) == 0

    def test_ingests_new_file(self):
        """New file (not in state) should be ingested."""
        state = MockStateReader()  # Empty state

        differ = Differ(state)

        scanned = [
            make_scanned_file("/root/new.md", content_hash="sha256:new"),
        ]

        result = differ.compute_diff(scanned)

        assert len(result.to_ingest) == 1
        assert len(result.to_skip) == 0

    def test_state_used_for_deletion_detection(self):
        """Files in state but not on disk should be marked deleted."""
        state = MockStateReader(
            active_paths={"/root/a.md", "/root/deleted.md"}
        )

        differ = Differ(state)

        # Only a.md is on disk now
        scanned = [make_scanned_file("/root/a.md")]

        result = differ.compute_diff(scanned)

        # deleted.md should be marked for deletion
        assert result.to_mark_deleted == ["/root/deleted.md"]

    def test_force_mode_ingests_everything(self):
        """Force mode should ingest everything regardless of state."""
        state = MockStateReader(
            active_paths={"/root/existing.md"},
            file_entries={
                "/root/existing.md": MockFileEntry("sha256:exists"),
            },
        )

        differ = Differ(state)

        scanned = [make_scanned_file("/root/existing.md", content_hash="sha256:exists")]

        result = differ.compute_diff(scanned, force=True)

        # Should be ingested even though hash matches
        assert len(result.to_ingest) == 1
        assert len(result.to_skip) == 0

    def test_empty_scan(self):
        """Empty scan results should produce empty diff."""
        state = MockStateReader()

        differ = Differ(state)
        result = differ.compute_diff([])

        assert len(result.to_ingest) == 0
        assert len(result.to_skip) == 0
        assert len(result.to_mark_deleted) == 0

    def test_all_files_new(self):
        """All new files should be ingested."""
        state = MockStateReader()

        differ = Differ(state)

        scanned = [
            make_scanned_file("/root/a.md", content_hash="sha256:a"),
            make_scanned_file("/root/b.md", content_hash="sha256:b"),
            make_scanned_file("/root/c.md", content_hash="sha256:c"),
        ]

        result = differ.compute_diff(scanned)

        assert len(result.to_ingest) == 3
        assert len(result.to_skip) == 0

    def test_all_files_unchanged(self):
        """All unchanged files should be skipped."""
        state = MockStateReader(
            active_paths={"/root/a.md", "/root/b.md", "/root/c.md"},
            file_entries={
                "/root/a.md": MockFileEntry("sha256:a"),
                "/root/b.md": MockFileEntry("sha256:b"),
                "/root/c.md": MockFileEntry("sha256:c"),
            },
        )

        differ = Differ(state)

        scanned = [
            make_scanned_file("/root/a.md", content_hash="sha256:a"),
            make_scanned_file("/root/b.md", content_hash="sha256:b"),
            make_scanned_file("/root/c.md", content_hash="sha256:c"),
        ]

        result = differ.compute_diff(scanned)

        assert len(result.to_ingest) == 0
        assert len(result.to_skip) == 3

    def test_uses_config_ids_from_state(self):
        """Config IDs should come from state reader."""
        state = MockStateReader(
            parser_id="custom_parser.v2",
            chunker_id="custom_1000_200",
            embedding_id="custom:embedding",
        )

        differ = Differ(state)

        scanned = [make_scanned_file("/root/test.md")]
        result = differ.compute_diff(scanned)

        candidate = result.to_ingest[0]
        assert candidate.parser_id == "custom_parser.v2"
        assert candidate.chunker_id == "custom_1000_200"
        assert candidate.embedding_id == "custom:embedding"

    def test_multiple_deletions(self):
        """Multiple deleted files should all be detected."""
        state = MockStateReader(
            active_paths={
                "/root/kept.md",
                "/root/deleted1.md",
                "/root/deleted2.md",
                "/root/deleted3.md",
            }
        )

        differ = Differ(state)

        # Only kept.md is on disk
        scanned = [make_scanned_file("/root/kept.md")]

        result = differ.compute_diff(scanned)

        assert len(result.to_mark_deleted) == 3
        assert "/root/deleted1.md" in result.to_mark_deleted
        assert "/root/deleted2.md" in result.to_mark_deleted
        assert "/root/deleted3.md" in result.to_mark_deleted


class TestComputeDiff:
    """Tests for compute_diff convenience function."""

    def test_computes_diff(self):
        """Test the convenience function."""
        state = MockStateReader(
            active_paths={"/root/existing.md"},
            file_entries={
                "/root/existing.md": MockFileEntry("sha256:exists"),
            },
        )

        scanned = [
            make_scanned_file("/root/existing.md", content_hash="sha256:exists"),
            make_scanned_file("/root/new.md", content_hash="sha256:new"),
        ]

        result = compute_diff(scanned, state_reader=state)

        assert len(result.to_skip) == 1
        assert len(result.to_ingest) == 1