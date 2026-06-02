# tests/unit/test_progressive_manifest.py
"""Unit tests for FileManifest — thread-safe manifest with JSON persistence."""

from __future__ import annotations

import threading
from pathlib import Path

from fitz_sage.engines.fitz_krag.progressive.manifest import (
    FileManifest,
    FileState,
    ManifestEntry,
    ManifestHeading,
    ManifestSymbol,
    indexing_status,
)


def _make_entry(
    rel_path: str = "src/main.py",
    *,
    file_id: str = "abc123",
    abs_path: str = "/project/src/main.py",
    content_hash: str = "sha256:deadbeef",
    file_type: str = ".py",
    size_bytes: int = 4300,
    state: FileState = FileState.REGISTERED,
    symbols: list[ManifestSymbol] | None = None,
    headings: list[ManifestHeading] | None = None,
    priority: int = 4,
    last_queried_at: float | None = None,
) -> ManifestEntry:
    return ManifestEntry(
        file_id=file_id,
        rel_path=rel_path,
        abs_path=abs_path,
        content_hash=content_hash,
        file_type=file_type,
        size_bytes=size_bytes,
        state=state,
        symbols=symbols or [],
        headings=headings or [],
        priority=priority,
        last_queried_at=last_queried_at,
    )


def test_indexing_status_counts_by_state(tmp_path: Path) -> None:
    """indexing_status() reports totals, pending, and a per-state breakdown."""
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_make_entry("a.py", file_id="a", state=FileState.REGISTERED))
    manifest.add(_make_entry("b.py", file_id="b", state=FileState.PARSED))
    manifest.add(_make_entry("c.py", file_id="c", state=FileState.KEYWORDED))
    manifest.add(_make_entry("d.py", file_id="d", state=FileState.QUERY_READY))
    manifest.add(_make_entry("e.py", file_id="e", state=FileState.ENRICHED))
    manifest.add(_make_entry("f.py", file_id="f", state=FileState.SUMMARIZED))

    status = indexing_status(manifest)

    assert status["total"] == 6
    assert status["pending"] == 3  # registered + parsed + keyworded
    assert status["indexed"] == 3  # query_ready + enriched + summarized
    assert status["complete"] is False
    assert status["query_ready"] is False
    assert status["deep_pending"] == 4
    assert status["fully_enriched"] is False
    assert status["by_state"] == {
        "registered": 1,
        "parsed": 1,
        "keyworded": 1,
        "query_ready": 1,
        "enriched": 1,
        "summarized": 1,
    }


def test_indexing_status_complete_when_all_query_ready(tmp_path: Path) -> None:
    """complete=True once every file can serve retrieval queries."""
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_make_entry("a.py", file_id="a", state=FileState.QUERY_READY))
    manifest.add(_make_entry("b.py", file_id="b", state=FileState.SUMMARIZED))

    status = indexing_status(manifest)

    assert status["total"] == 2
    assert status["pending"] == 0
    assert status["complete"] is True
    assert status["query_ready"] is True
    assert status["deep_pending"] == 1
    assert status["fully_enriched"] is False


def test_indexing_status_fully_enriched_when_deep_work_done(tmp_path: Path) -> None:
    """fully_enriched=True once all files completed mandatory deep enrichment."""
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_make_entry("a.py", file_id="a", state=FileState.ENRICHED))
    manifest.add(_make_entry("b.py", file_id="b", state=FileState.SUMMARIZED))

    status = indexing_status(manifest)

    assert status["complete"] is True
    assert status["deep_pending"] == 0
    assert status["fully_enriched"] is True


def test_indexing_status_none_manifest() -> None:
    """A None manifest reports an empty, complete status."""
    assert indexing_status(None) == {
        "total": 0,
        "indexed": 0,
        "pending": 0,
        "complete": True,
        "query_ready": True,
        "deep_pending": 0,
        "fully_enriched": True,
        "by_state": {},
    }


class TestFileManifest:
    """Tests for FileManifest operations."""

    def test_add_and_get(self, tmp_path: Path) -> None:
        """Add an entry then retrieve it by rel_path."""
        manifest = FileManifest(tmp_path / "manifest.json")
        entry = _make_entry("lib/utils.py", file_id="u1")

        manifest.add(entry)
        result = manifest.get("lib/utils.py")

        assert result is not None
        assert result.file_id == "u1"
        assert result.rel_path == "lib/utils.py"
        assert result.state == FileState.REGISTERED

        # Non-existent key returns None
        assert manifest.get("does/not/exist.py") is None

    def test_update_state(self, tmp_path: Path) -> None:
        """Transition REGISTERED -> PARSED preserves all other fields."""
        manifest = FileManifest(tmp_path / "manifest.json")
        sym = ManifestSymbol("main", "main", "function", "args", 10, 25)
        entry = _make_entry(
            "app.py",
            file_id="s1",
            size_bytes=9000,
            symbols=[sym],
            priority=2,
        )
        manifest.add(entry)

        manifest.update_state("app.py", FileState.PARSED)
        updated = manifest.get("app.py")

        assert updated is not None
        assert updated.state == FileState.PARSED
        # All other fields preserved
        assert updated.file_id == "s1"
        assert updated.size_bytes == 9000
        assert updated.priority == 2
        assert len(updated.symbols) == 1
        assert updated.symbols[0].name == "main"

    def test_files_in_state(self, tmp_path: Path) -> None:
        """Filter entries by matching state."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("a.py", state=FileState.REGISTERED))
        manifest.add(_make_entry("b.py", state=FileState.PARSED))
        manifest.add(_make_entry("c.py", state=FileState.REGISTERED))
        manifest.add(_make_entry("d.py", state=FileState.SUMMARIZED))

        registered = manifest.files_in_state(FileState.REGISTERED)
        parsed = manifest.files_in_state(FileState.PARSED)
        summarized = manifest.files_in_state(FileState.SUMMARIZED)

        assert len(registered) == 2
        assert {e.rel_path for e in registered} == {"a.py", "c.py"}
        assert len(parsed) == 1
        assert parsed[0].rel_path == "b.py"
        assert len(summarized) == 1
        assert summarized[0].rel_path == "d.py"

    def test_files_not_in_state(self, tmp_path: Path) -> None:
        """Inverse filter: entries NOT at a given state."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("a.py", state=FileState.SUMMARIZED))
        manifest.add(_make_entry("b.py", state=FileState.SUMMARIZED))
        manifest.add(_make_entry("c.py", state=FileState.PARSED))

        not_summarized = manifest.files_not_in_state(FileState.SUMMARIZED)

        assert len(not_summarized) == 1
        assert not_summarized[0].rel_path == "c.py"

    def test_files_not_query_ready(self, tmp_path: Path) -> None:
        """Query-ready filtering includes all later enrichment states."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("a.py", state=FileState.PARSED))
        manifest.add(_make_entry("b.py", state=FileState.KEYWORDED))
        manifest.add(_make_entry("c.py", state=FileState.QUERY_READY))
        manifest.add(_make_entry("d.py", state=FileState.ENRICHED))

        not_ready = manifest.files_not_query_ready()

        assert {entry.rel_path for entry in not_ready} == {"a.py", "b.py"}

    def test_bump_priority_sets_p1(self, tmp_path: Path) -> None:
        """Queried files become priority 1 with a last_queried_at timestamp."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("hot.py", priority=4))
        manifest.add(_make_entry("cold.py", priority=4))

        manifest.bump_priority(["hot.py"])

        hot = manifest.get("hot.py")
        cold = manifest.get("cold.py")
        assert hot is not None
        assert hot.priority == 1
        assert hot.last_queried_at is not None
        assert cold is not None
        assert cold.priority == 4
        assert cold.last_queried_at is None

    def test_bump_priority_level_only_improves(self, tmp_path: Path) -> None:
        """Priority only improves (lower number). P3->P2 succeeds, P1->P3 rejected."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("mid.py", priority=3))
        manifest.add(_make_entry("top.py", priority=1))

        manifest.bump_priority_level(["mid.py", "top.py"], level=2)

        mid = manifest.get("mid.py")
        top = manifest.get("top.py")
        assert mid is not None
        assert mid.priority == 2  # 3 -> 2 (improved)
        assert top is not None
        assert top.priority == 1  # 1 -> 2 rejected (would worsen)

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        """Save manifest to JSON then load into a fresh instance; entries preserved."""
        path = tmp_path / "sub" / "manifest.json"
        m1 = FileManifest(path)
        sym = ManifestSymbol("greet", "mod.greet", "function", "name: str", 5, 12)
        heading = ManifestHeading("Overview", 1)
        m1.add(
            _make_entry(
                "mod.py",
                file_id="r1",
                content_hash="sha256:aabb",
                size_bytes=2048,
                state=FileState.SUMMARIZED,
                symbols=[sym],
                priority=2,
                last_queried_at=1700000000.0,
            )
        )
        m1.add(
            _make_entry(
                "docs/guide.md",
                file_id="r2",
                file_type=".md",
                size_bytes=1100,
                state=FileState.SUMMARIZED,
                headings=[heading],
            )
        )
        m1.save()

        # Load into a fresh FileManifest (constructor auto-loads when file exists)
        m2 = FileManifest(path)

        assert len(m2.entries()) == 2

        py_entry = m2.get("mod.py")
        assert py_entry is not None
        assert py_entry.file_id == "r1"
        assert py_entry.state == FileState.SUMMARIZED
        assert py_entry.priority == 2
        assert py_entry.last_queried_at == 1700000000.0
        assert len(py_entry.symbols) == 1
        assert py_entry.symbols[0].name == "greet"
        assert py_entry.symbols[0].signature == "name: str"

        md_entry = m2.get("docs/guide.md")
        assert md_entry is not None
        assert md_entry.state == FileState.SUMMARIZED
        assert len(md_entry.headings) == 1
        assert md_entry.headings[0].title == "Overview"
        assert md_entry.headings[0].level == 1

    def test_to_manifest_text_symbols(self, tmp_path: Path) -> None:
        """Code file with symbols renders compact symbol lines."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(
            _make_entry(
                "src/engine.py",
                size_bytes=4300,
                file_type=".py",
                symbols=[
                    ManifestSymbol("main", "engine.main", "function", "args", 10, 25),
                    ManifestSymbol("Engine", "engine.Engine", "class", None, 30, 80),
                ],
            )
        )

        text = manifest.to_manifest_text()

        assert "src/engine.py [4.2KB, python]" in text
        assert "  fun: main(args) L10-25" in text
        assert "  cla: Engine L30-80" in text

    def test_to_manifest_text_headings(self, tmp_path: Path) -> None:
        """Document file with headings renders markdown-style heading lines."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(
            _make_entry(
                "docs/guide.md",
                size_bytes=1100,
                file_type=".md",
                headings=[
                    ManifestHeading("Introduction", 1),
                    ManifestHeading("Getting Started", 2),
                ],
            )
        )

        text = manifest.to_manifest_text()

        assert "docs/guide.md [1.1KB, markdown]" in text
        assert "  # Introduction" in text
        assert "  ## Getting Started" in text

    def test_clear(self, tmp_path: Path) -> None:
        """Clear removes all entries."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("a.py"))
        manifest.add(_make_entry("b.py"))
        assert len(manifest.entries()) == 2

        manifest.clear()

        assert len(manifest.entries()) == 0
        assert manifest.get("a.py") is None

    def test_entries_returns_snapshot(self, tmp_path: Path) -> None:
        """Modifying the returned dict does not affect the manifest's internal state."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("keep.py"))

        snapshot = manifest.entries()
        snapshot["keep.py"] = _make_entry("keep.py", file_id="tampered")
        snapshot["injected.py"] = _make_entry("injected.py")

        # Internal state unchanged
        original = manifest.get("keep.py")
        assert original is not None
        assert original.file_id == "abc123"  # not "tampered"
        assert manifest.get("injected.py") is None
        assert len(manifest.entries()) == 1

    def test_concurrent_add_update(self, tmp_path: Path) -> None:
        """Multiple threads adding and updating entries do not corrupt state."""
        manifest = FileManifest(tmp_path / "manifest.json")
        num_threads = 20
        entries_per_thread = 50
        barrier = threading.Barrier(num_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(entries_per_thread):
                rel = f"t{thread_id}/f{i}.py"
                manifest.add(_make_entry(rel, file_id=f"{thread_id}-{i}"))
            # Update state on own entries
            for i in range(entries_per_thread):
                rel = f"t{thread_id}/f{i}.py"
                manifest.update_state(rel, FileState.PARSED)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_entries = manifest.entries()
        assert len(all_entries) == num_threads * entries_per_thread

        # Every entry should have been transitioned to PARSED
        for entry in all_entries.values():
            assert entry.state == FileState.PARSED
