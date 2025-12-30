# tests/test_epistemic_hierarchy.py
"""Tests for epistemic-aware hierarchical summarization.

Note: Conflict detection at ingest time requires a semantic matcher with
an embedder. The stub find_conflicts() function returns empty without one.

For actual conflict detection, see test_constraints.py which uses
SemanticMatcher with mock embedder.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_ai.core.chunk import Chunk
from fitz_ai.ingestion.enrichment.config import HierarchyConfig
from fitz_ai.ingestion.enrichment.hierarchy import (
    EpistemicAssessment,
    HierarchyEnricher,
    assess_chunk_group,
)
from fitz_ai.prompts.hierarchy import (
    build_epistemic_corpus_prompt,
    build_epistemic_group_prompt,
)


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

    def test_conflicting_classifications_detected(self):
        """Test that conflicting classifications are detected via the constraint plugin."""
        from fitz_ai.core.guardrails import SemanticMatcher
        from tests.conftest_guardrails import create_deterministic_embedder

        embedder = create_deterministic_embedder()
        semantic_matcher = SemanticMatcher(embedder=embedder)

        # Use the same conflict patterns that ConflictAwareConstraint detects
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

        assessment = assess_chunk_group(chunks, semantic_matcher=semantic_matcher)
        assert assessment.has_conflicts
        assert assessment.agreement_ratio < 1.0

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
        config = HierarchyConfig(enabled=True, group_by="source")

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "This is a summary."

        enricher = HierarchyEnricher(config=config, chat_client=mock_chat)

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

    def test_conflict_detection_uses_constraint_plugin(self):
        """Test that conflict detection uses the existing ConflictAwareConstraint logic."""
        from fitz_ai.core.guardrails import SemanticMatcher
        from tests.conftest_guardrails import create_deterministic_embedder

        config = HierarchyConfig(enabled=True, group_by="source")

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "Summary acknowledging conflicts."

        embedder = create_deterministic_embedder()
        semantic_matcher = SemanticMatcher(embedder=embedder)

        enricher = HierarchyEnricher(
            config=config,
            chat_client=mock_chat,
            semantic_matcher=semantic_matcher,
        )

        # Create chunks with conflicting classifications (what ConflictAwareConstraint detects)
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

        # L1 summaries are now stored as metadata on original chunks (level 0)
        level0_chunks = [c for c in result if c.metadata.get("hierarchy_level") == 0]
        assert len(level0_chunks) == 2

        # All original chunks should have the same L1 summary and detect conflict
        for chunk in level0_chunks:
            assert chunk.metadata.get("epistemic_has_conflicts") is True

    def test_corpus_summary_includes_epistemic_aggregates(self):
        """Test that corpus summary has aggregated epistemic metadata."""
        config = HierarchyConfig(enabled=True, group_by="source")

        mock_chat = MagicMock()
        mock_chat.chat.return_value = "Summary content."

        enricher = HierarchyEnricher(config=config, chat_client=mock_chat)

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

    def test_disabled_enricher_skips_epistemic(self):
        """Test that disabled enricher returns chunks unchanged."""
        config = HierarchyConfig(enabled=False)

        mock_chat = MagicMock()
        enricher = HierarchyEnricher(config=config, chat_client=mock_chat)

        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                content="Some content.",
                chunk_index=0,
                metadata={},
            ),
        ]

        result = enricher.enrich(chunks)

        assert len(result) == 1
        assert result[0].id == "c1"
        mock_chat.chat.assert_not_called()


class TestSingleSourceOfTruth:
    """Tests verifying that epistemic detection uses the existing constraint plugins."""

    def test_uses_conflict_aware_constraint_patterns(self):
        """Verify that the same patterns detected by ConflictAwareConstraint are detected here."""
        from fitz_ai.core.guardrails import ConflictAwareConstraint, SemanticMatcher
        from tests.conftest_guardrails import create_deterministic_embedder

        # Create semantic matcher for both constraint and assessment
        embedder = create_deterministic_embedder()
        semantic_matcher = SemanticMatcher(embedder=embedder)

        # Create chunks with a known conflict pattern
        chunks = [
            Chunk(
                id="c1",
                doc_id="d1",
                content="The status is confirmed and verified.",
                chunk_index=0,
                metadata={},
            ),
            Chunk(
                id="c2",
                doc_id="d1",
                content="The status remains unconfirmed pending review.",
                chunk_index=1,
                metadata={},
            ),
        ]

        # Check via constraint directly
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)
        constraint_result = constraint.apply("What is the status?", chunks)

        # Check via epistemic assessment with semantic matcher
        assessment = assess_chunk_group(chunks, semantic_matcher=semantic_matcher)

        # Both should agree on whether there are conflicts
        constraint_found_conflict = not constraint_result.allow_decisive_answer
        assessment_found_conflict = assessment.has_conflicts

        assert constraint_found_conflict == assessment_found_conflict
