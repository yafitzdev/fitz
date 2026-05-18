# tests/unit/test_progressive_worker.py
"""Unit tests for BackgroundIngestWorker — the progressive ingestion scheduler.

The worker is a scheduler over the KragIngestPipeline core: it owns the
manifest, priority queue, state machine, and query-pausing, and delegates all
ingestion work to ``core.parse_file`` / ``summarize_file`` / ``enrich_file`` /
``finalize``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.progressive.manifest import FileManifest, FileState, ManifestEntry
from fitz_sage.engines.fitz_krag.progressive.worker import BackgroundIngestWorker


def _make_entry(
    rel_path: str,
    priority: int = 4,
    size_bytes: int = 1000,
    state: FileState = FileState.REGISTERED,
    file_type: str | None = None,
) -> ManifestEntry:
    """Build a ManifestEntry for testing."""
    ext = file_type if file_type is not None else Path(rel_path).suffix
    return ManifestEntry(
        file_id=f"id-{rel_path}",
        rel_path=rel_path,
        abs_path=f"/fake/{rel_path}",
        content_hash="abc123",
        file_type=ext,
        size_bytes=size_bytes,
        state=state,
        symbols=[],
        headings=[],
        priority=priority,
    )


def _build_worker(
    manifest: object | None = None,
    core: MagicMock | None = None,
    source_dir: Path = Path("/fake"),
) -> BackgroundIngestWorker:
    """Construct a BackgroundIngestWorker with a mocked core."""
    if manifest is None:
        manifest = MagicMock()
    return BackgroundIngestWorker(
        manifest=manifest,
        source_dir=source_dir,
        core=core or MagicMock(),
    )


# -------------------------------------------------------------------------
# 1. _get_ordered_files sorts by (priority, size_bytes)
# -------------------------------------------------------------------------


class TestGetOrderedFiles:
    def test_get_ordered_files_priority(self) -> None:
        """P1 files come before P2, which come before P4."""
        p4_big = _make_entry("a/large.py", priority=4, size_bytes=5000)
        p1_small = _make_entry("b/hot.py", priority=1, size_bytes=200)
        p2_med = _make_entry("c/sibling.py", priority=2, size_bytes=3000)

        manifest = MagicMock()
        manifest.files_in_state.return_value = [p4_big, p1_small, p2_med]

        worker = _build_worker(manifest=manifest)
        ordered = worker._get_ordered_files(FileState.REGISTERED)

        assert ordered[0].rel_path == "b/hot.py", "P1 should be first"
        assert ordered[1].rel_path == "c/sibling.py", "P2 should be second"
        assert ordered[2].rel_path == "a/large.py", "P4 should be last"

    def test_same_priority_sorted_by_size(self) -> None:
        """Within the same priority, smaller files come first."""
        big = _make_entry("big.py", priority=4, size_bytes=9999)
        small = _make_entry("small.py", priority=4, size_bytes=100)

        manifest = MagicMock()
        manifest.files_in_state.return_value = [big, small]

        worker = _build_worker(manifest=manifest)
        ordered = worker._get_ordered_files(FileState.REGISTERED)

        assert ordered[0].rel_path == "small.py"
        assert ordered[1].rel_path == "big.py"


# -------------------------------------------------------------------------
# 2. _run drives the core through every phase
# -------------------------------------------------------------------------


class TestScheduling:
    def test_run_drives_core_through_all_phases(self, tmp_path: Path) -> None:
        """Every REGISTERED file is parsed, summarized, enriched; finalize runs once."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("src/main.py", file_type=".py"))
        manifest.add(_make_entry("docs/readme.md", file_type=".md"))

        core = MagicMock()
        worker = _build_worker(manifest=manifest, core=core, source_dir=tmp_path)
        worker._run()

        # Both files reached the terminal ENRICHED state
        assert manifest.get("src/main.py").state == FileState.ENRICHED
        assert manifest.get("docs/readme.md").state == FileState.ENRICHED

        # Core ops called once per file, finalize once for the corpus
        assert core.parse_file.call_count == 2
        assert core.summarize_file.call_count == 2
        assert core.enrich_file.call_count == 2
        core.finalize.assert_called_once()

    def test_parse_phase_passes_file_identity_to_core(self, tmp_path: Path) -> None:
        """The worker hands the core (rel_path, abs_path, file_id) for each file."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("src/main.py", file_type=".py"))

        core = MagicMock()
        worker = _build_worker(manifest=manifest, core=core, source_dir=tmp_path)
        worker._parse_phase()

        core.parse_file.assert_called_once()
        rel_path, _abs_path, file_id = core.parse_file.call_args[0]
        assert rel_path == "src/main.py"
        assert file_id == "id-src/main.py"
        assert manifest.get("src/main.py").state == FileState.PARSED

    def test_summarize_and_enrich_route_by_file_type(self, tmp_path: Path) -> None:
        """summarize_file / enrich_file receive the file's id and type."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("a.py", file_type=".py", state=FileState.PARSED))

        core = MagicMock()
        worker = _build_worker(manifest=manifest, core=core, source_dir=tmp_path)
        worker._summarize_phase()
        core.summarize_file.assert_called_once_with("id-a.py", ".py")
        assert manifest.get("a.py").state == FileState.SUMMARIZED

        worker._enrich_phase()
        core.enrich_file.assert_called_once_with("id-a.py", ".py")
        assert manifest.get("a.py").state == FileState.ENRICHED

    def test_parse_failure_does_not_block_other_files(self, tmp_path: Path) -> None:
        """One file failing to parse leaves it behind but does not stop the rest."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("bad.py", file_type=".py"))
        manifest.add(_make_entry("good.py", file_type=".py"))

        core = MagicMock()

        def _parse(rel_path, _abs_path, _file_id):
            if rel_path == "bad.py":
                raise RuntimeError("parse blew up")

        core.parse_file.side_effect = _parse

        worker = _build_worker(manifest=manifest, core=core, source_dir=tmp_path)
        worker._parse_phase()

        assert manifest.get("bad.py").state == FileState.REGISTERED
        assert manifest.get("good.py").state == FileState.PARSED

    def test_stop_event_halts_processing(self, tmp_path: Path) -> None:
        """A set stop event prevents any core work."""
        manifest = FileManifest(tmp_path / "manifest.json")
        manifest.add(_make_entry("src/app.py", file_type=".py"))

        core = MagicMock()
        worker = _build_worker(manifest=manifest, core=core, source_dir=tmp_path)
        worker._stop_event.set()

        worker._run()

        core.parse_file.assert_not_called()
        core.finalize.assert_not_called()
        assert manifest.get("src/app.py").state == FileState.REGISTERED


# -------------------------------------------------------------------------
# 3. boost_files sets P1 on queried files, P2 on directory siblings
# -------------------------------------------------------------------------


class TestBoostFiles:
    def test_boost_files_sets_p1_and_siblings_p2(self) -> None:
        """boost_files bumps queried files to P1 and same-dir siblings to P2."""
        queried = _make_entry("src/main.py", state=FileState.PARSED, priority=4)
        sibling = _make_entry("src/utils.py", state=FileState.PARSED, priority=4)
        unrelated = _make_entry("docs/readme.md", state=FileState.PARSED, priority=4)
        already_done = _make_entry("src/done.py", state=FileState.ENRICHED, priority=4)

        manifest = MagicMock()
        manifest.entries.return_value = {
            "src/main.py": queried,
            "src/utils.py": sibling,
            "docs/readme.md": unrelated,
            "src/done.py": already_done,
        }

        worker = _build_worker(manifest=manifest)
        worker.boost_files(["src/main.py"])

        # Queried file bumped to P1
        manifest.bump_priority.assert_called_once_with(["src/main.py"])

        # Sibling in same dir (src/) bumped to P2 — but NOT the queried file
        # itself, NOT an unrelated dir, NOT an already-ENRICHED file
        manifest.bump_priority_level.assert_called_once()
        call_args = manifest.bump_priority_level.call_args
        siblings_arg = call_args[0][0]
        level_arg = call_args[1]["level"] if "level" in call_args[1] else call_args[0][1]

        assert "src/utils.py" in siblings_arg
        assert "src/main.py" not in siblings_arg, "queried file should not be in siblings"
        assert "docs/readme.md" not in siblings_arg, "unrelated dir should not be in siblings"
        assert "src/done.py" not in siblings_arg, "ENRICHED files should be excluded"
        assert level_arg == 2
