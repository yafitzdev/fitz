# tests/unit/test_krag_hierarchy.py
"""
Unit tests for L1/L2 hierarchy generation in the KRAG ingestion core.

These exercise the live enrichment path: the background worker schedules
``core.build_hierarchy_file`` and ``core.build_corpus_hierarchy``. This is the
path ``engine.point()`` runs in production — see test_progressive_worker.py
for the worker → core scheduling, and these tests for what each op produces.

Tests that hierarchy output is persisted and model failures are surfaced to
the worker for separate enrichment-state reporting.

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
    chat: MagicMock | None = None,
):
    """Create a KragIngestPipeline core with a mocked connection manager."""
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

    config = FitzKragConfig(collection="test_col")
    return KragIngestPipeline(
        config=config,
        chat=chat or MagicMock(),
        connection_manager=MagicMock(),
        collection="test_col",
    )


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
        core = _make_core(chat=chat)
        core._section_store = MagicMock()
        core._section_store.get_by_file.return_value = _section_dicts(3)

        core.build_hierarchy_file("file-1", ".md")

        # Enrichment persisted once, carrying the L1 summary in metadata
        core._section_store.update_entities_by_file.assert_called_once()
        persisted = core._section_store.update_entities_by_file.call_args[0][1]
        for sec in persisted:
            assert sec["metadata"]["hierarchy_summary"] == "Document covers setup instructions."

    def test_l1_failure_is_surfaced(self):
        """The worker must see L1 failures so it can report enrichment failure."""
        import pytest

        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("Timeout")
        core = _make_core(chat=chat)
        core._section_store = MagicMock()
        core._section_store.get_by_file.return_value = _section_dicts(3)

        with pytest.raises(RuntimeError, match="Timeout"):
            core.build_hierarchy_file("file-1", ".md")
        core._section_store.update_entities_by_file.assert_not_called()


# ---------------------------------------------------------------------------
# TestL2CorpusSummary — produced by core.finalize
# ---------------------------------------------------------------------------


class TestL2CorpusSummary:
    """L2 corpus summary, built and stored by the corpus finalize step."""

    def test_finalize_stores_l2_corpus_summary(self):
        """finalize rolls L1 summaries into an L2 summary stored as a section."""
        from fitz_sage.engines.fitz_krag.ingestion.section_store import (
            CORPUS_SUMMARY_SCHEMA_VERSION,
        )

        chat = MagicMock()
        chat.chat.return_value = "This corpus documents the system architecture."
        core = _make_core(chat=chat)
        core._section_store = MagicMock()
        core._section_store.get_hierarchy_summaries.return_value = ["L1 of doc A", "L1 of doc B"]
        core._raw_store = MagicMock()
        core._import_store = MagicMock()

        core.build_corpus_hierarchy()

        # L2 stored under a synthetic raw file + a retrievable corpus section
        core._raw_store.upsert.assert_called_once()
        core._section_store.upsert_batch.assert_called_once()
        stored = core._section_store.upsert_batch.call_args[0][0]
        assert len(stored) == 1
        assert stored[0]["metadata"]["is_corpus_summary"] is True
        assert stored[0]["metadata"]["corpus_summary_schema"] == CORPUS_SUMMARY_SCHEMA_VERSION
        assert stored[0]["metadata"]["source_signature"]
        assert stored[0]["content"] == "This corpus documents the system architecture."
        core._section_store.delete_by_file.assert_called_once_with("__krag_corpus__")
        core._raw_store.delete.assert_called_once_with("__krag_corpus__")

    def test_finalize_skips_l2_without_l1_summaries(self):
        """No L1 summaries means no L2 corpus summary is stored."""
        core = _make_core(chat=MagicMock())
        core._section_store = MagicMock()
        core._section_store.get_hierarchy_summaries.return_value = []
        core._raw_store = MagicMock()
        core._import_store = MagicMock()

        core.build_corpus_hierarchy()

        core._section_store.upsert_batch.assert_not_called()
        core._section_store.delete_by_file.assert_called_once_with("__krag_corpus__")
        core._raw_store.delete.assert_called_once_with("__krag_corpus__")

    def test_l2_failure_is_surfaced(self):
        """Collection failure is surfaced for persisted finalization status."""
        import pytest

        chat = MagicMock()
        chat.chat.side_effect = RuntimeError("Timeout")
        core = _make_core(chat=chat)
        core._section_store = MagicMock()
        core._section_store.get_hierarchy_summaries.return_value = ["L1 of doc A"]
        core._raw_store = MagicMock()
        core._import_store = MagicMock()

        with pytest.raises(RuntimeError, match="Timeout"):
            core.build_corpus_hierarchy()

        core._section_store.upsert_batch.assert_not_called()
        core._section_store.delete_by_file.assert_called_once_with("__krag_corpus__")
        core._raw_store.delete.assert_called_once_with("__krag_corpus__")
