# tests/unit/test_krag_hierarchy.py
"""
Unit tests for L1/L2 hierarchy generation in the KRAG ingestion core.

These exercise the *live* ingestion path: the background worker schedules
``core.enrich_file`` (per file) and ``core.finalize`` (corpus). This is the
path ``engine.point()`` runs in production — see test_progressive_worker.py
for the worker → core scheduling, and these tests for what each op produces.

Tests that:
- enrich_file adds an L1 hierarchy_summary to each section's metadata
- finalize generates and stores the L2 corpus summary as a section
- hierarchy is skipped when enable_hierarchy=False
- LLM errors fail gracefully

Code symbols deliberately have no hierarchy stage — they already carry
machine-readable structure (imports, AST), so symbol-level summaries are
redundant for code.
"""

from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_core(
    *,
    enable_hierarchy: bool = True,
    enable_enrichment: bool = True,
    chat: MagicMock | None = None,
):
    """Create a KragIngestPipeline core with a mocked connection manager."""
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

    config = FitzKragConfig(
        collection="test_col",
        enable_enrichment=enable_enrichment,
        enable_hierarchy=enable_hierarchy,
    )
    return KragIngestPipeline(
        config=config,
        chat=chat or MagicMock(),
        connection_manager=MagicMock(),
        collection="test_col",
    )


def _fake_enricher() -> MagicMock:
    """Enricher stub that stamps keywords/entities onto dicts in place (no LLM)."""
    enricher = MagicMock()

    def _stamp(dicts):
        for d in dicts:
            d["keywords"] = ["alpha"]
            d["entities"] = []

    enricher.enrich_symbols.side_effect = _stamp
    enricher.enrich_sections.side_effect = _stamp
    return enricher


def _section_dicts(count: int = 3, file_id: str = "file-1") -> list[dict]:
    """Section dicts shaped like SectionStore.get_by_file output."""
    return [
        {
            "id": f"sec-{i}",
            "raw_file_id": file_id,
            "title": f"Section {i}",
            "level": 1,
            "content": f"Content of section {i}",
            "summary": f"Summary of section {i}",
            "keywords": [],
            "entities": [],
            "metadata": {},
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# TestL1SectionHierarchy — produced by core.enrich_file
# ---------------------------------------------------------------------------


class TestL1SectionHierarchy:
    """L1 file-level summaries, stamped onto section metadata during enrich."""

    def test_enrich_file_adds_l1_summary_to_sections(self):
        """enrich_file on a doc generates one L1 summary, stamped on every section."""
        chat = MagicMock()
        chat.chat.return_value = "Document covers setup instructions."
        core = _make_core(enable_hierarchy=True, chat=chat)
        core._enricher = _fake_enricher()
        core._section_store = MagicMock()
        core._section_store.get_by_file.return_value = _section_dicts(3)

        core.enrich_file("file-1", ".md")

        # Enrichment persisted once, carrying the L1 summary in metadata
        core._section_store.update_enrichment_by_file.assert_called_once()
        persisted = core._section_store.update_enrichment_by_file.call_args[0][1]
        for sec in persisted:
            assert sec["metadata"]["hierarchy_summary"] == "Document covers setup instructions."

    def test_enrich_file_skips_l1_when_hierarchy_disabled(self):
        """enable_hierarchy=False means no hierarchy_summary on section metadata."""
        chat = MagicMock()
        chat.chat.return_value = "unused"
        core = _make_core(enable_hierarchy=False, chat=chat)
        core._enricher = _fake_enricher()
        core._section_store = MagicMock()
        core._section_store.get_by_file.return_value = _section_dicts(2)

        core.enrich_file("file-1", ".md")

        persisted = core._section_store.update_enrichment_by_file.call_args[0][1]
        for sec in persisted:
            assert "hierarchy_summary" not in sec["metadata"]

    def test_l1_failure_does_not_crash(self):
        """An LLM failure during L1 generation is caught; enrichment still persists."""
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("Timeout")
        core = _make_core(enable_hierarchy=True, chat=chat)
        core._enricher = _fake_enricher()
        core._section_store = MagicMock()
        core._section_store.get_by_file.return_value = _section_dicts(3)

        # Should not raise
        core.enrich_file("file-1", ".md")

        persisted = core._section_store.update_enrichment_by_file.call_args[0][1]
        for sec in persisted:
            assert "hierarchy_summary" not in sec["metadata"]

    def test_l1_runs_without_enricher(self):
        """L1 hierarchy is produced even when keyword/entity enrichment is disabled.

        enable_hierarchy and enable_enrichment are independent flags.
        """
        chat = MagicMock()
        chat.chat.return_value = "Document overview."
        core = _make_core(enable_hierarchy=True, enable_enrichment=False, chat=chat)
        assert core._enricher is None
        core._section_store = MagicMock()
        core._section_store.get_by_file.return_value = _section_dicts(2)

        core.enrich_file("file-1", ".md")

        core._section_store.update_enrichment_by_file.assert_called_once()
        persisted = core._section_store.update_enrichment_by_file.call_args[0][1]
        for sec in persisted:
            assert sec["metadata"]["hierarchy_summary"] == "Document overview."


# ---------------------------------------------------------------------------
# TestL2CorpusSummary — produced by core.finalize
# ---------------------------------------------------------------------------


class TestL2CorpusSummary:
    """L2 corpus summary, built and stored by the corpus finalize step."""

    def test_finalize_stores_l2_corpus_summary(self):
        """finalize rolls L1 summaries into an L2 summary stored as a section."""
        chat = MagicMock()
        chat.chat.return_value = "This corpus documents the system architecture."
        core = _make_core(enable_hierarchy=True, chat=chat)
        core._section_store = MagicMock()
        core._section_store.get_hierarchy_summaries.return_value = ["L1 of doc A", "L1 of doc B"]
        core._raw_store = MagicMock()
        core._import_store = MagicMock()

        core.finalize()

        # L2 stored under a synthetic raw file + a retrievable corpus section
        core._raw_store.upsert.assert_called_once()
        core._section_store.upsert_batch.assert_called_once()
        stored = core._section_store.upsert_batch.call_args[0][0]
        assert len(stored) == 1
        assert stored[0]["metadata"]["is_corpus_summary"] is True
        assert stored[0]["content"] == "This corpus documents the system architecture."

    def test_finalize_skips_l2_without_l1_summaries(self):
        """No L1 summaries means no L2 corpus summary is stored."""
        core = _make_core(enable_hierarchy=True, chat=MagicMock())
        core._section_store = MagicMock()
        core._section_store.get_hierarchy_summaries.return_value = []
        core._raw_store = MagicMock()
        core._import_store = MagicMock()

        core.finalize()

        core._section_store.upsert_batch.assert_not_called()

    def test_finalize_skips_l2_when_hierarchy_disabled(self):
        """enable_hierarchy=False: finalize resolves imports but builds no L2."""
        core = _make_core(enable_hierarchy=False, chat=MagicMock())
        core._section_store = MagicMock()
        core._raw_store = MagicMock()
        core._import_store = MagicMock()

        core.finalize()

        core._section_store.get_hierarchy_summaries.assert_not_called()
        core._section_store.upsert_batch.assert_not_called()
        # Import resolution still runs regardless of hierarchy config
        core._import_store.resolve_targets.assert_called_once()

    def test_l2_failure_does_not_crash(self):
        """An LLM failure during L2 generation is caught; nothing is stored."""
        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("Timeout")
        core = _make_core(enable_hierarchy=True, chat=chat)
        core._section_store = MagicMock()
        core._section_store.get_hierarchy_summaries.return_value = ["L1 of doc A"]
        core._raw_store = MagicMock()
        core._import_store = MagicMock()

        # Should not raise
        core.finalize()

        core._section_store.upsert_batch.assert_not_called()
