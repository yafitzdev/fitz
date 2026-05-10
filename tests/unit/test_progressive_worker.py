# tests/unit/test_progressive_worker.py
"""Unit tests for BackgroundIngestWorker."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.progressive.manifest import FileState, ManifestEntry
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
    manifest: MagicMock | None = None,
    enricher: MagicMock | None = MagicMock(),
    symbol_store: MagicMock | None = None,
    section_store: MagicMock | None = None,
) -> BackgroundIngestWorker:
    """Construct a BackgroundIngestWorker with fully mocked dependencies."""
    if manifest is None:
        manifest = MagicMock()
        manifest._path = Path("/fake/manifest.json")

    if symbol_store is None:
        symbol_store = MagicMock()
    if section_store is None:
        section_store = MagicMock()

    return BackgroundIngestWorker(
        manifest=manifest,
        source_dir=Path("/fake"),
        config=MagicMock(),
        chat=MagicMock(),
        connection_manager=MagicMock(),
        collection="test",
        stores={
            "raw": MagicMock(),
            "symbol": symbol_store,
            "import": MagicMock(),
            "section": section_store,
            "table": MagicMock(),
        },
        enricher=enricher,
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
        manifest._path = Path("/fake/manifest.json")
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
        manifest._path = Path("/fake/manifest.json")
        manifest.files_in_state.return_value = [big, small]

        worker = _build_worker(manifest=manifest)
        ordered = worker._get_ordered_files(FileState.REGISTERED)

        assert ordered[0].rel_path == "small.py"
        assert ordered[1].rel_path == "big.py"


# -------------------------------------------------------------------------
# 2. _enrich_summarized_files calls enricher for code files
# -------------------------------------------------------------------------


class TestEnrichPhase:
    def test_enrich_phase_calls_enricher(self) -> None:
        """enricher.enrich_symbols is called for a .py file in SUMMARIZED state."""
        py_entry = _make_entry("src/main.py", state=FileState.SUMMARIZED, file_type=".py")

        manifest = MagicMock()
        manifest._path = Path("/fake/manifest.json")
        manifest.files_in_state.return_value = [py_entry]

        mock_symbol_store = MagicMock()
        mock_symbol_store.get_by_file.return_value = [{"name": "foo", "kind": "function"}]

        mock_enricher = MagicMock()

        worker = _build_worker(
            manifest=manifest,
            enricher=mock_enricher,
            symbol_store=mock_symbol_store,
        )
        worker._enrich_summarized_files()

        mock_symbol_store.get_by_file.assert_called_once_with(py_entry.file_id)
        mock_enricher.enrich_symbols.assert_called_once()
        mock_symbol_store.update_enrichment_by_file.assert_called_once()


# -------------------------------------------------------------------------
# 3. _enrich_summarized_files does not crash when enricher is None
# -------------------------------------------------------------------------


class TestEnrichSkipsWhenNoEnricher:
    def test_enrich_skips_when_no_enricher(self) -> None:
        """When enricher=None, _enrich_summarized_files returns without error."""
        manifest = MagicMock()
        manifest._path = Path("/fake/manifest.json")
        manifest.files_in_state.return_value = [
            _make_entry("src/app.py", state=FileState.SUMMARIZED),
        ]

        worker = _build_worker(manifest=manifest, enricher=None)

        # Should not raise and should not call manifest.files_in_state
        worker._enrich_summarized_files()
        manifest.files_in_state.assert_not_called()


# -------------------------------------------------------------------------
# 4. Code vs doc routing: .py -> enrich_symbols, .md -> enrich_sections
# -------------------------------------------------------------------------


class TestEnrichCodeVsDocRouting:
    def test_enrich_code_vs_doc_routing(self) -> None:
        """.py routes to _enrich_file_symbols, .md routes to _enrich_file_sections."""
        py_entry = _make_entry("src/lib.py", state=FileState.SUMMARIZED, file_type=".py")
        md_entry = _make_entry("docs/readme.md", state=FileState.SUMMARIZED, file_type=".md")

        manifest = MagicMock()
        manifest._path = Path("/fake/manifest.json")
        manifest.files_in_state.return_value = [py_entry, md_entry]

        mock_symbol_store = MagicMock()
        mock_symbol_store.get_by_file.return_value = [{"name": "bar"}]

        mock_section_store = MagicMock()
        mock_section_store.get_by_file.return_value = [{"title": "Intro"}]

        mock_enricher = MagicMock()

        worker = _build_worker(
            manifest=manifest,
            enricher=mock_enricher,
            symbol_store=mock_symbol_store,
            section_store=mock_section_store,
        )
        worker._enrich_summarized_files()

        # .py -> enrich_symbols
        mock_enricher.enrich_symbols.assert_called_once()
        mock_symbol_store.update_enrichment_by_file.assert_called_once_with(
            py_entry.file_id, [{"name": "bar"}]
        )

        # .md -> enrich_sections
        mock_enricher.enrich_sections.assert_called_once()
        mock_section_store.update_enrichment_by_file.assert_called_once_with(
            md_entry.file_id, [{"title": "Intro"}]
        )


# -------------------------------------------------------------------------
# 5. .csv/.tsv files are skipped entirely in enrichment
# -------------------------------------------------------------------------


class TestEnrichSkipsTableFiles:
    def test_enrich_skips_table_files(self) -> None:
        """.csv and .tsv files are never enriched."""
        csv_entry = _make_entry("data/sales.csv", state=FileState.SUMMARIZED, file_type=".csv")
        tsv_entry = _make_entry("data/logs.tsv", state=FileState.SUMMARIZED, file_type=".tsv")

        manifest = MagicMock()
        manifest._path = Path("/fake/manifest.json")
        manifest.files_in_state.return_value = [csv_entry, tsv_entry]

        mock_enricher = MagicMock()
        mock_symbol_store = MagicMock()
        mock_section_store = MagicMock()

        worker = _build_worker(
            manifest=manifest,
            enricher=mock_enricher,
            symbol_store=mock_symbol_store,
            section_store=mock_section_store,
        )
        worker._enrich_summarized_files()

        mock_enricher.enrich_symbols.assert_not_called()
        mock_enricher.enrich_sections.assert_not_called()
        mock_symbol_store.get_by_file.assert_not_called()
        mock_section_store.get_by_file.assert_not_called()


# -------------------------------------------------------------------------
# 6. boost_files sets P1 on queried files, P2 on directory siblings
# -------------------------------------------------------------------------


class TestBoostFiles:
    def test_boost_files_sets_p1_and_siblings_p2(self) -> None:
        """boost_files bumps queried files to P1 and same-dir siblings to P2."""
        queried = _make_entry("src/main.py", state=FileState.PARSED, priority=4)
        sibling = _make_entry("src/utils.py", state=FileState.PARSED, priority=4)
        unrelated = _make_entry("docs/readme.md", state=FileState.PARSED, priority=4)
        already_done = _make_entry("src/done.py", state=FileState.EMBEDDED, priority=4)

        manifest = MagicMock()
        manifest._path = Path("/fake/manifest.json")
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

        # Sibling in same dir (src/) bumped to P2, but NOT the queried file itself,
        # NOT unrelated dir, NOT already-EMBEDDED file
        manifest.bump_priority_level.assert_called_once()
        call_args = manifest.bump_priority_level.call_args
        siblings_arg = call_args[0][0]
        level_arg = call_args[1]["level"] if "level" in call_args[1] else call_args[0][1]

        assert "src/utils.py" in siblings_arg
        assert "src/main.py" not in siblings_arg, "queried file should not be in siblings"
        assert "docs/readme.md" not in siblings_arg, "unrelated dir should not be in siblings"
        assert "src/done.py" not in siblings_arg, "EMBEDDED files should be excluded"
        assert level_arg == 2


# -------------------------------------------------------------------------
# 7. _parse_summary_response: valid JSON array
# -------------------------------------------------------------------------


class TestParseSummaryResponse:
    def test_parse_summary_response_json(self) -> None:
        """Valid JSON array is parsed correctly."""
        worker = _build_worker()
        response = json.dumps(["Summary A", "Summary B", "Summary C"])

        result = worker._parse_summary_response(response, 3)

        assert result == ["Summary A", "Summary B", "Summary C"]

    # ---------------------------------------------------------------------
    # 8. Invalid JSON falls back to line splitting
    # ---------------------------------------------------------------------

    def test_parse_summary_response_text_fallback(self) -> None:
        """Non-JSON text falls back to line splitting."""
        worker = _build_worker()
        response = "First summary line\nSecond summary line"

        result = worker._parse_summary_response(response, 2)

        assert result == ["First summary line", "Second summary line"]

    # ---------------------------------------------------------------------
    # 9. Code fence (```json ... ```) is stripped before parsing
    # ---------------------------------------------------------------------

    def test_parse_summary_response_code_fence(self) -> None:
        """Response wrapped in ```json fences is stripped and parsed as JSON."""
        worker = _build_worker()
        inner = json.dumps(["Alpha", "Beta"])
        response = f"```json\n{inner}\n```"

        result = worker._parse_summary_response(response, 2)

        assert result == ["Alpha", "Beta"]

    def test_parse_summary_response_pads_missing(self) -> None:
        """If fewer lines than expected, pad with '(no summary)'."""
        worker = _build_worker()
        response = "Only one line"

        result = worker._parse_summary_response(response, 3)

        assert len(result) == 3
        assert result[0] == "Only one line"
        assert result[1] == "(no summary)"
        assert result[2] == "(no summary)"

    def test_parse_summary_response_truncates_excess(self) -> None:
        """JSON array longer than expected_count is truncated."""
        worker = _build_worker()
        response = json.dumps(["A", "B", "C", "D", "E"])

        result = worker._parse_summary_response(response, 3)

        assert result == ["A", "B", "C"]


# -------------------------------------------------------------------------
# 10. _stop_event causes immediate return from enrichment
# -------------------------------------------------------------------------


class TestStopEvent:
    def test_stop_event_skips_enrichment(self) -> None:
        """When _stop_event is set, _enrich_summarized_files returns immediately."""
        py_entry = _make_entry("src/app.py", state=FileState.SUMMARIZED, file_type=".py")

        manifest = MagicMock()
        manifest._path = Path("/fake/manifest.json")
        manifest.files_in_state.return_value = [py_entry]

        mock_enricher = MagicMock()
        mock_symbol_store = MagicMock()

        worker = _build_worker(
            manifest=manifest,
            enricher=mock_enricher,
            symbol_store=mock_symbol_store,
        )

        # Set stop before calling
        worker._stop_event.set()

        worker._enrich_summarized_files()

        # The enricher should never be called because stop_event triggers early return
        mock_enricher.enrich_symbols.assert_not_called()
        mock_symbol_store.get_by_file.assert_not_called()
