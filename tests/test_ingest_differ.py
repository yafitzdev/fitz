# tests/test_ingest_differ.py
"""
Tests for fitz_ai.ingest.diff.differ module.

Key tests verify that:
1. State file is used for skip decisions (authoritative source)
2. Differ only computes actions, doesn't execute
3. Deletion detection works via state comparison
4. Config changes trigger re-ingestion
"""

from typing import Dict, Optional, Set

from fitz_ai.ingest.diff.differ import (
    ConfigProvider,
    Differ,
    DiffResult,
    FileCandidate,
    ReingestReason,
    StateReader,
    compute_diff,
)
from fitz_ai.ingest.diff.scanner import ScannedFile


class MockFileEntry:
    """Mock file entry for testing."""

    def __init__(
        self,
        content_hash: str,
        chunker_id: str = "simple:1000:0",
        parser_id: str = "md.v1",
        embedding_id: str = "cohere:embed-english-v3.0",
    ):
        self.content_hash = content_hash
        self.chunker_id = chunker_id
        self.parser_id = parser_id
        self.embedding_id = embedding_id


class MockStateReader:
    """Mock state reader for testing."""

    def __init__(
        self,
        active_paths: Optional[Set[str]] = None,
        file_entries: Optional[Dict[str, MockFileEntry]] = None,
    ):
        self._active_paths = active_paths or set()
        self._file_entries = file_entries or {}

    def get_active_paths(self, root: str) -> Set[str]:
        return self._active_paths

    def get_file_entry(self, root: str, file_path: str) -> Optional[MockFileEntry]:
        return self._file_entries.get(file_path)


class MockConfigProvider:
    """Mock config provider (like ChunkingRouter)."""

    def __init__(
        self,
        chunker_id: str = "simple:1000:0",
        chunker_ids_by_ext: Optional[Dict[str, str]] = None,
    ):
        self._default = chunker_id
        self._by_ext = chunker_ids_by_ext or {}

    def get_chunker_id(self, ext: str) -> str:
        return self._by_ext.get(ext, self._default)


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


def make_differ(
    state_reader: Optional[StateReader] = None,
    config_provider: Optional[ConfigProvider] = None,
    parser_id_func: Optional[callable] = None,
    embedding_id: str = "cohere:embed-english-v3.0",
) -> Differ:
    """Helper to create a Differ with defaults."""
    return Differ(
        state_reader=state_reader or MockStateReader(),
        config_provider=config_provider or MockConfigProvider(),
        parser_id_func=parser_id_func or (lambda ext: f"{ext.lstrip('.')}.v1"),
        embedding_id=embedding_id,
    )


class TestFileCandidate:
    """Tests for FileCandidate dataclass."""

    def test_from_scanned(self):
        """Test creating FileCandidate from ScannedFile."""
        scanned = make_scanned_file("/root/test.md")

        candidate = FileCandidate.from_scanned(
            scanned,
            parser_id="md.v1",
            chunker_id="simple:1000:0",
            embedding_id="cohere:embed-english-v3.0",
        )

        assert candidate.path == "/root/test.md"
        assert candidate.content_hash == "sha256:abc123"
        assert candidate.parser_id == "md.v1"
        assert candidate.chunker_id == "simple:1000:0"
        assert candidate.embedding_id == "cohere:embed-english-v3.0"


class TestDiffResult:
    """Tests for DiffResult dataclass."""

    def test_summary(self):
        """Test summary property."""
        result = DiffResult(
            to_ingest=[
                FileCandidate.from_scanned(
                    make_scanned_file("/root/a.md"),
                    "md.v1",
                    "simple:1000:0",
                    "cohere:embed-english-v3.0",
                )
            ],
            to_skip=[
                FileCandidate.from_scanned(
                    make_scanned_file("/root/b.md"),
                    "md.v1",
                    "simple:1000:0",
                    "cohere:embed-english-v3.0",
                )
            ],
            to_mark_deleted=["/root/c.md", "/root/d.md"],
        )

        assert result.summary == "ingest=1, skip=1, delete=2"


class TestDiffer:
    """Tests for Differ class."""

    def test_skips_file_with_matching_hash_in_state(self):
        """File should be skipped if state has entry with same hash and config."""
        state = MockStateReader(
            active_paths={"/root/existing.md"},
            file_entries={
                "/root/existing.md": MockFileEntry("sha256:exists"),
            },
        )

        differ = make_differ(state_reader=state)

        scanned = [
            make_scanned_file("/root/existing.md", content_hash="sha256:exists"),
        ]

        result = differ.compute_diff(scanned)

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

        differ = make_differ(state_reader=state)

        scanned = [
            make_scanned_file("/root/changed.md", content_hash="sha256:new_hash"),
        ]

        result = differ.compute_diff(scanned)

        assert len(result.to_ingest) == 1
        assert len(result.to_skip) == 0

    def test_ingests_new_file(self):
        """New file (not in state) should be ingested."""
        state = MockStateReader()

        differ = make_differ(state_reader=state)

        scanned = [
            make_scanned_file("/root/new.md", content_hash="sha256:new"),
        ]

        result = differ.compute_diff(scanned)

        assert len(result.to_ingest) == 1
        assert len(result.to_skip) == 0

    def test_state_used_for_deletion_detection(self):
        """Files in state but not on disk should be marked deleted."""
        state = MockStateReader(active_paths={"/root/a.md", "/root/deleted.md"})

        differ = make_differ(state_reader=state)

        scanned = [make_scanned_file("/root/a.md")]

        result = differ.compute_diff(scanned)

        assert result.to_mark_deleted == ["/root/deleted.md"]

    def test_force_mode_ingests_everything(self):
        """Force mode should ingest everything regardless of state."""
        state = MockStateReader(
            active_paths={"/root/existing.md"},
            file_entries={
                "/root/existing.md": MockFileEntry("sha256:exists"),
            },
        )

        differ = make_differ(state_reader=state)

        scanned = [make_scanned_file("/root/existing.md", content_hash="sha256:exists")]

        result = differ.compute_diff(scanned, force=True)

        assert len(result.to_ingest) == 1
        assert len(result.to_skip) == 0

    def test_empty_scan(self):
        """Empty scan results should produce empty diff."""
        state = MockStateReader()

        differ = make_differ(state_reader=state)
        result = differ.compute_diff([])

        assert len(result.to_ingest) == 0
        assert len(result.to_skip) == 0
        assert len(result.to_mark_deleted) == 0

    def test_all_files_new(self):
        """All new files should be ingested."""
        state = MockStateReader()

        differ = make_differ(state_reader=state)

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

        differ = make_differ(state_reader=state)

        scanned = [
            make_scanned_file("/root/a.md", content_hash="sha256:a"),
            make_scanned_file("/root/b.md", content_hash="sha256:b"),
            make_scanned_file("/root/c.md", content_hash="sha256:c"),
        ]

        result = differ.compute_diff(scanned)

        assert len(result.to_ingest) == 0
        assert len(result.to_skip) == 3

    def test_uses_config_ids_from_providers(self):
        """Config IDs should come from config provider."""
        state = MockStateReader()
        config = MockConfigProvider(chunker_id="custom:500:50")

        differ = make_differ(
            state_reader=state,
            config_provider=config,
            parser_id_func=lambda ext: "custom_parser.v2",
            embedding_id="custom:embedding",
        )

        scanned = [make_scanned_file("/root/test.md")]
        result = differ.compute_diff(scanned)

        candidate = result.to_ingest[0]
        assert candidate.chunker_id == "custom:500:50"
        assert candidate.parser_id == "custom_parser.v2"
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

        differ = make_differ(state_reader=state)

        scanned = [make_scanned_file("/root/kept.md")]

        result = differ.compute_diff(scanned)

        assert len(result.to_mark_deleted) == 3
        assert "/root/deleted1.md" in result.to_mark_deleted
        assert "/root/deleted2.md" in result.to_mark_deleted
        assert "/root/deleted3.md" in result.to_mark_deleted

    def test_chunker_id_change_triggers_reingest(self):
        """File should be re-ingested when chunker_id changes."""
        state = MockStateReader(
            active_paths={"/root/doc.md"},
            file_entries={
                "/root/doc.md": MockFileEntry(
                    content_hash="sha256:same",
                    chunker_id="simple:1000:0",
                ),
            },
        )

        config = MockConfigProvider(chunker_id="simple:500:50")

        differ = make_differ(state_reader=state, config_provider=config)

        scanned = [make_scanned_file("/root/doc.md", content_hash="sha256:same")]
        result = differ.compute_diff(scanned)

        assert len(result.to_ingest) == 1
        assert len(result.to_skip) == 0


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
        config = MockConfigProvider()

        scanned = [
            make_scanned_file("/root/existing.md", content_hash="sha256:exists"),
            make_scanned_file("/root/new.md", content_hash="sha256:new"),
        ]

        result = compute_diff(
            scanned,
            state_reader=state,
            config_provider=config,
            parser_id_func=lambda ext: "md.v1",
            embedding_id="cohere:embed-english-v3.0",
        )

        assert len(result.to_skip) == 1
        assert len(result.to_ingest) == 1


class TestReingestReason:
    """Tests for ReingestReason dataclass."""

    def test_new_file(self):
        """New file has is_new=True."""
        reason = ReingestReason(is_new=True)
        assert reason.needs_reingest
        assert "new" in str(reason)

    def test_content_changed(self):
        """Content change sets content_changed=True."""
        reason = ReingestReason(content_changed=True)
        assert reason.needs_reingest
        assert "content_changed" in str(reason)

    def test_chunker_changed(self):
        """Chunker change sets chunker_changed=True."""
        reason = ReingestReason(chunker_changed=True)
        assert reason.needs_reingest
        assert "chunker_changed" in str(reason)

    def test_no_change(self):
        """No changes means no reingest needed."""
        reason = ReingestReason()
        assert not reason.needs_reingest
        assert str(reason) == "none"
