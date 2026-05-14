# tests/test_epistemic_hierarchy.py
"""Tests for epistemic-aware hierarchical summarization.

Note: Conflict detection at ingest time is deferred - the stub find_conflicts()
returns empty. Actual conflict detection happens at query time using LLM-based
analysis via the pyrrho classifier.

For conflict detection tests, see test_governance_constraints.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.core.chunk import Chunk
from fitz_sage.ingestion.enrichment.config import HierarchyConfig
from fitz_sage.ingestion.enrichment.hierarchy import (
    EpistemicAssessment,
    HierarchyEnricher,
    assess_chunk_group,
)
from fitz_sage.prompts.hierarchy import (
    build_epistemic_corpus_prompt,
    build_epistemic_group_prompt,
)


def create_mock_chat_factory(mock_chat):
    """Create a mock chat factory that returns the mock chat client."""

    def factory(tier: str = "fast"):
        return mock_chat

    return factory


class TestEpistemicAssessment:
    """Tests for the EpistemicAssessment dataclass and assess_chunk_group function."""

    def test_empty_chunks_returns_sparse(self):
        assessment = assess_chunk_group([])
        assert assessment.evidence_density == "sparse"
        assert assessment.chunk_count == 0
        assert not assessment.has_conflicts

    def test_single_chunk_no_conflicts(self):
        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                content="NPS improved to 65 this quarter.",
                chunk_index=0,
                metadata={},
            ),
        ]

        assessment = assess_chunk_group(chunks)
        assert not assessment.has_conflicts
        assert assessment.chunk_count == 1
        assert assessment.evidence_density == "sparse"

    def test_ingest_time_conflict_detection_deferred(self):
        """Conflict detection during ingestion is deferred to query time
        (handled now by the pyrrho classifier).
        """
        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                content="This was classified as a security incident.",
                chunk_index=0,
                metadata={},
            ),
            Chunk(
                id="c2",
                doc_id="d1",
                content="This was classified as an operational incident.",
                chunk_index=1,
                metadata={},
            ),
        ]

        assessment = assess_chunk_group(chunks)
        assert assessment.has_conflicts is False
        assert assessment.agreement_ratio == 1.0

    def test_evidence_density_sparse(self):
        chunks = [
            Chunk(id="c1", doc_id="d1", content="Short.", chunk_index=0, metadata={}),
        ]

        assessment = assess_chunk_group(chunks)
        assert assessment.evidence_density == "sparse"

    def test_evidence_density_moderate(self):
        chunks = [
            Chunk(
                id=f"c{i}",
                doc_id="d1",
                content="Some content here. " * 50,
                chunk_index=i,
                metadata={},
            )
            for i in range(5)
        ]

        assessment = assess_chunk_group(chunks)
        assert assessment.evidence_density == "moderate"

    def test_evidence_density_dense(self):
        chunks = [
            Chunk(
                id=f"c{i}",
                doc_id="d1",
                content="Substantial content here. " * 100,
                chunk_index=i,
                metadata={},
            )
            for i in range(15)
        ]

        assessment = assess_chunk_group(chunks)
        assert assessment.evidence_density == "dense"

    def test_to_metadata(self):
        assessment = EpistemicAssessment(
            has_conflicts=True,
            conflict_topics=["classification"],
            agreement_ratio=0.6,
            evidence_density="moderate",
            chunk_count=5,
        )

        meta = assessment.to_metadata()
        assert meta["epistemic_has_conflicts"] is True
        assert meta["epistemic_conflict_topics"] == ["classification"]
        assert meta["epistemic_agreement_ratio"] == 0.6
        assert meta["epistemic_evidence_density"] == "moderate"


class TestEpistemicPrompts:
    """Tests for epistemic-aware prompt generation."""

    def test_group_prompt_with_conflicts(self):
        assessment = EpistemicAssessment(
            has_conflicts=True,
            conflict_topics=["classification"],
            agreement_ratio=0.6,
            evidence_density="moderate",
            chunk_count=5,
        )

        prompt = build_epistemic_group_prompt(assessment)
        assert "CONFLICTING SOURCES DETECTED" in prompt
        assert "classification" in prompt
        assert "Do NOT pick a side" in prompt

    def test_group_prompt_with_sparse_evidence(self):
        assessment = EpistemicAssessment(
            has_conflicts=False,
            evidence_density="sparse",
            chunk_count=2,
        )

        prompt = build_epistemic_group_prompt(assessment)
        assert "LIMITED EVIDENCE" in prompt
        assert "sparse" in prompt

    def test_group_prompt_with_low_agreement(self):
        assessment = EpistemicAssessment(
            has_conflicts=False,
            agreement_ratio=0.5,
            evidence_density="moderate",
            chunk_count=10,
        )

        prompt = build_epistemic_group_prompt(assessment)
        assert "LOW SOURCE AGREEMENT" in prompt
        assert "50%" in prompt

    def test_group_prompt_no_issues(self):
        assessment = EpistemicAssessment(
            has_conflicts=False,
            agreement_ratio=0.9,
            evidence_density="dense",
            chunk_count=20,
        )

        prompt = build_epistemic_group_prompt(assessment)
        # Should be the base prompt without warnings
        assert "CONFLICTING SOURCES" not in prompt
        assert "LIMITED EVIDENCE" not in prompt
        assert "LOW SOURCE AGREEMENT" not in prompt

    def test_corpus_prompt_with_group_conflicts(self):
        assessments = [
            EpistemicAssessment(has_conflicts=True, conflict_topics=["classification"]),
            EpistemicAssessment(has_conflicts=False),
            EpistemicAssessment(has_conflicts=True, conflict_topics=["classification"]),
        ]

        prompt = build_epistemic_corpus_prompt(assessments)
        assert "GROUP SUMMARIES CONTAIN CONFLICTS" in prompt
        assert "2 of 3" in prompt  # 2 groups with conflicts

    def test_corpus_prompt_with_sparse_groups(self):
        assessments = [
            EpistemicAssessment(evidence_density="sparse"),
            EpistemicAssessment(evidence_density="sparse"),
            EpistemicAssessment(evidence_density="moderate"),
        ]

        prompt = build_epistemic_corpus_prompt(assessments)
        assert "LIMITED COVERAGE" in prompt


class TestHierarchyEnricherEpistemic:
    """Tests for HierarchyEnricher with epistemic features."""

    def test_simple_mode_adds_epistemic_metadata(self):
        """Test that simple mode adds epistemic metadata to original chunks."""
        config = HierarchyConfig(group_by="source", min_group_chunks=1, min_group_content=0)

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "This is a summary."

        enricher = HierarchyEnricher(
            config=config, chat_factory=create_mock_chat_factory(mock_chat)
        )

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                content="NPS improved to 65.",
                chunk_index=0,
                metadata={"source": "file1.txt"},
            ),
            Chunk(
                id="c2",
                doc_id="d1",
                content="Customer satisfaction also improved.",
                chunk_index=1,
                metadata={"source": "file1.txt"},
            ),
        ]

        result = enricher.enrich(chunks)

        # L1 summaries are now stored as metadata on original chunks, not as separate chunks
        # Find original chunks (level 0)
        level0_chunks = [c for c in result if c.metadata.get("hierarchy_level") == 0]
        assert len(level0_chunks) == 2

        # Each original chunk should have the L1 summary as metadata
        for chunk in level0_chunks:
            assert "hierarchy_summary" in chunk.metadata
            assert "epistemic_has_conflicts" in chunk.metadata
            assert "epistemic_evidence_density" in chunk.metadata
            assert "epistemic_agreement_ratio" in chunk.metadata

        # Should also have one L2 corpus summary chunk
        level2_chunks = [c for c in result if c.metadata.get("hierarchy_level") == 2]
        assert len(level2_chunks) == 1

    def test_enricher_adds_epistemic_metadata(self):
        """Enricher adds epistemic metadata (evidence density, chunk count).
        Conflict detection itself is deferred to query time (pyrrho).
        """
        config = HierarchyConfig(group_by="source", min_group_chunks=1, min_group_content=0)

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "Summary of the incident reports."

        enricher = HierarchyEnricher(
            config=config,
            chat_factory=create_mock_chat_factory(mock_chat),
        )

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                content="This is a security incident that affected production.",
                chunk_index=0,
                metadata={"source": "report1.txt"},
            ),
            Chunk(
                id="c2",
                doc_id="d1",
                content="This is an operational incident, not security related.",
                chunk_index=1,
                metadata={"source": "report1.txt"},
            ),
        ]

        result = enricher.enrich(chunks)

        # L1 summaries are stored as metadata on original chunks (level 0)
        level0_chunks = [c for c in result if c.metadata.get("hierarchy_level") == 0]
        assert len(level0_chunks) == 2

        # Enricher adds epistemic metadata but doesn't detect conflicts at ingest time
        for chunk in level0_chunks:
            assert "epistemic_has_conflicts" in chunk.metadata
            assert "epistemic_evidence_density" in chunk.metadata
            # Conflicts are not detected at ingest time (deferred to query time)
            assert chunk.metadata.get("epistemic_has_conflicts") is False

    def test_corpus_summary_includes_epistemic_aggregates(self):
        """Test that corpus summary has aggregated epistemic metadata."""
        config = HierarchyConfig(group_by="source", min_group_chunks=1, min_group_content=0)

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "Summary content."

        enricher = HierarchyEnricher(
            config=config, chat_factory=create_mock_chat_factory(mock_chat)
        )

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                content="Content for file 1.",
                chunk_index=0,
                metadata={"source": "file1.txt"},
            ),
            Chunk(
                id="c2",
                doc_id="d1",
                content="Content for file 2.",
                chunk_index=1,
                metadata={"source": "file2.txt"},
            ),
        ]

        result = enricher.enrich(chunks)

        # Find the corpus summary (level 2)
        corpus_summaries = [c for c in result if c.metadata.get("hierarchy_level") == 2]
        assert len(corpus_summaries) == 1

        corpus = corpus_summaries[0]
        assert "epistemic_groups_with_conflicts" in corpus.metadata
        assert "epistemic_evidence_density" in corpus.metadata
        assert "epistemic_agreement_ratio" in corpus.metadata
