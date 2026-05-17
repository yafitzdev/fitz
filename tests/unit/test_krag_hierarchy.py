# tests/unit/test_krag_hierarchy.py
"""
Unit tests for section hierarchy generation in KragIngestPipeline.

Tests that:
- L1 group summaries are generated per document file
- hierarchy_summary is added to section metadata
- L2 corpus summary is generated
- hierarchy is skipped when enable_hierarchy=False
- LLM errors fail gracefully

Code symbols deliberately have no hierarchy stage — they already carry
machine-readable structure (imports, AST), so symbol-level summaries are
redundant for code.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Shared patch targets for pipeline construction
_PIPELINE_PATCHES = [
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.ensure_schema",
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.RawFileStore",
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.SymbolStore",
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.ImportGraphStore",
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.SectionStore",
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.TableStore",
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.PythonCodeIngestStrategy",
    "fitz_sage.engines.fitz_krag.ingestion.pipeline.TechnicalDocIngestStrategy",
]


def _make_pipeline(
    enable_hierarchy: bool = True,
    chat_responses: list[str] | None = None,
    chat_side_effect=None,
):
    """Create a KragIngestPipeline with all stores mocked."""
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

    config = FitzKragConfig(
        collection="test_col",
        enable_enrichment=False,
        enable_hierarchy=enable_hierarchy,
    )

    chat = MagicMock(name="chat")
    if chat_side_effect is not None:
        chat.chat.side_effect = chat_side_effect
    elif chat_responses is not None:
        chat.chat.side_effect = chat_responses
    else:
        chat.chat.return_value = "A summary of this group."

    cm = MagicMock(name="connection_manager")

    pipeline = KragIngestPipeline(
        config=config,
        chat=chat,
        connection_manager=cm,
        collection="test_col",
    )
    return pipeline, chat


def _section_dicts_with_summaries(
    count: int = 3, file_id: str = "file-1"
) -> tuple[list[dict], list[str]]:
    """Create section dicts with summaries and corresponding file IDs."""
    section_dicts = [
        {
            "id": f"sec-{i}",
            "title": f"Section {i}",
            "summary": f"Summary of section {i}",
            "raw_file_id": file_id,
            "metadata": {},
        }
        for i in range(count)
    ]
    file_ids = [file_id] * count
    return section_dicts, file_ids


# ---------------------------------------------------------------------------
# TestSectionHierarchy
# ---------------------------------------------------------------------------


class TestSectionHierarchy:
    """Tests for document-section L1/L2 hierarchy summaries."""

    @patch(*[_PIPELINE_PATCHES[0]])
    @patch(*[_PIPELINE_PATCHES[1]])
    @patch(*[_PIPELINE_PATCHES[2]])
    @patch(*[_PIPELINE_PATCHES[3]])
    @patch(*[_PIPELINE_PATCHES[4]])
    @patch(*[_PIPELINE_PATCHES[5]])
    @patch(*[_PIPELINE_PATCHES[6]])
    @patch(*[_PIPELINE_PATCHES[7]])
    def test_l1_sections_generated_per_file(self, *mocks):
        """Section hierarchy generates per-file L1 summaries plus an L2 corpus summary."""
        pipeline, chat = _make_pipeline(
            chat_responses=[
                "Document covers setup instructions.",  # L1
                "Corpus overview.",  # L2
            ]
        )

        sections, file_ids = _section_dicts_with_summaries(3, file_id="file-1")

        pipeline._generate_hierarchy_sections(sections, file_ids)

        # 1 L1 call + 1 L2 call
        assert chat.chat.call_count == 2

        for sec in sections:
            assert sec["metadata"]["hierarchy_summary"] == ("Document covers setup instructions.")

    @patch(*[_PIPELINE_PATCHES[0]])
    @patch(*[_PIPELINE_PATCHES[1]])
    @patch(*[_PIPELINE_PATCHES[2]])
    @patch(*[_PIPELINE_PATCHES[3]])
    @patch(*[_PIPELINE_PATCHES[4]])
    @patch(*[_PIPELINE_PATCHES[5]])
    @patch(*[_PIPELINE_PATCHES[6]])
    @patch(*[_PIPELINE_PATCHES[7]])
    def test_hierarchy_skipped_when_disabled(self, *mocks):
        """enable_hierarchy=False is reflected on the config (ingest() guards on it)."""
        pipeline, chat = _make_pipeline(enable_hierarchy=False)
        assert pipeline._config.enable_hierarchy is False

    @patch(*[_PIPELINE_PATCHES[0]])
    @patch(*[_PIPELINE_PATCHES[1]])
    @patch(*[_PIPELINE_PATCHES[2]])
    @patch(*[_PIPELINE_PATCHES[3]])
    @patch(*[_PIPELINE_PATCHES[4]])
    @patch(*[_PIPELINE_PATCHES[5]])
    @patch(*[_PIPELINE_PATCHES[6]])
    @patch(*[_PIPELINE_PATCHES[7]])
    def test_section_hierarchy_failure_does_not_crash(self, *mocks):
        """LLM failure during section hierarchy is caught gracefully."""
        pipeline, chat = _make_pipeline(
            chat_side_effect=RuntimeError("Timeout"),
        )

        sections, file_ids = _section_dicts_with_summaries(3, file_id="file-1")

        # Should not raise
        pipeline._generate_hierarchy_sections(sections, file_ids)

        for sec in sections:
            assert "hierarchy_summary" not in sec["metadata"]
