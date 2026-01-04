# tests/test_semantic_grouping.py
"""Tests for semantic grouping module."""

from __future__ import annotations

import pytest

# Skip all tests in this module if sklearn is not installed
# Must be before any imports that use sklearn
pytest.importorskip("sklearn", reason="sklearn required for semantic grouping tests")

from unittest.mock import MagicMock

import numpy as np

from fitz_ai.core.chunk import Chunk
from fitz_ai.ingestion.enrichment import EnrichmentConfig, EnrichmentPipeline
from fitz_ai.ingestion.enrichment.config import HierarchyConfig
from fitz_ai.ingestion.enrichment.hierarchy import (
    EmbeddingProvider,
    HierarchyEnricher,
    SemanticGrouper,
)


class TestSemanticGrouper:
    """Tests for SemanticGrouper class."""

    def test_group_basic(self):
        """Test basic semantic grouping."""
        grouper = SemanticGrouper(n_clusters=2)

        # Create chunks
        chunks = [
            Chunk(id="c1", doc_id="d1", chunk_index=0, content="Topic A content 1", metadata={}),
            Chunk(id="c2", doc_id="d1", chunk_index=1, content="Topic A content 2", metadata={}),
            Chunk(id="c3", doc_id="d2", chunk_index=0, content="Topic B content 1", metadata={}),
            Chunk(id="c4", doc_id="d2", chunk_index=1, content="Topic B content 2", metadata={}),
        ]

        # Create embeddings that cluster into 2 groups
        # First two similar, last two similar
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Topic A
                [0.9, 0.1, 0.0],  # Topic A (similar)
                [0.0, 1.0, 0.0],  # Topic B
                [0.1, 0.9, 0.0],  # Topic B (similar)
            ],
            dtype=np.float32,
        )

        groups = grouper.group(chunks, embeddings)

        assert len(groups) == 2
        # Each group should have 2 chunks
        for group_key, group_chunks in groups.items():
            assert len(group_chunks) == 2
            assert group_key.startswith("cluster_")

    def test_group_auto_k(self):
        """Test automatic k detection."""
        grouper = SemanticGrouper(n_clusters=None, max_clusters=5)

        chunks = [
            Chunk(id=f"c{i}", doc_id="d1", chunk_index=i, content=f"Content {i}", metadata={})
            for i in range(10)
        ]

        # Create embeddings with 3 distinct clusters
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.2, 0.8, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
                [0.2, 0.0, 0.8],
                [0.3, 0.0, 0.7],
            ],
            dtype=np.float32,
        )

        groups = grouper.group(chunks, embeddings)

        # Should auto-detect some reasonable number of clusters
        assert 2 <= len(groups) <= 5

    def test_group_empty(self):
        """Test grouping with empty input."""
        grouper = SemanticGrouper(n_clusters=3)
        groups = grouper.group([], np.array([]).reshape(0, 0))
        assert groups == {}

    def test_group_single_chunk(self):
        """Test grouping with single chunk."""
        grouper = SemanticGrouper(n_clusters=None)
        chunks = [Chunk(id="c1", doc_id="d1", chunk_index=0, content="Content", metadata={})]
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        groups = grouper.group(chunks, embeddings)

        # Should put single chunk in one group
        assert len(groups) == 1
        assert "cluster_0" in groups
        assert len(groups["cluster_0"]) == 1

    def test_group_mismatch_raises(self):
        """Test that mismatched chunks/embeddings raises error."""
        grouper = SemanticGrouper(n_clusters=2)
        chunks = [
            Chunk(id="c1", doc_id="d1", chunk_index=0, content="Content", metadata={}),
            Chunk(id="c2", doc_id="d1", chunk_index=1, content="Content 2", metadata={}),
        ]
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)  # Only 1 embedding

        with pytest.raises(ValueError, match="Mismatch"):
            grouper.group(chunks, embeddings)


class TestEmbeddingProvider:
    """Tests for EmbeddingProvider class."""

    def test_get_embeddings(self):
        """Test embedding computation."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]

        provider = EmbeddingProvider(mock_embedder)

        chunks = [
            Chunk(id="c1", doc_id="d1", chunk_index=0, content="Content 1", metadata={}),
            Chunk(id="c2", doc_id="d1", chunk_index=1, content="Content 2", metadata={}),
        ]

        embeddings = provider.get_embeddings(chunks)

        assert embeddings.shape == (2, 3)
        assert mock_embedder.embed.call_count == 2

    def test_get_embeddings_empty(self):
        """Test embedding computation with empty input."""
        mock_embedder = MagicMock()
        provider = EmbeddingProvider(mock_embedder)

        embeddings = provider.get_embeddings([])

        assert embeddings.shape == (0, 0)
        mock_embedder.embed.assert_not_called()


class TestHierarchyConfigSemantic:
    """Tests for semantic grouping configuration."""

    def test_config_defaults(self):
        """Test default config values."""
        config = HierarchyConfig()
        assert config.grouping_strategy == "metadata"
        assert config.n_clusters is None
        assert config.max_clusters == 10

    def test_config_from_dict(self):
        """Test config from dict."""
        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "hierarchy": {
                    "enabled": True,
                    "grouping_strategy": "semantic",
                    "n_clusters": 5,
                    "max_clusters": 15,
                },
            }
        )

        assert config.hierarchy.grouping_strategy == "semantic"
        assert config.hierarchy.n_clusters == 5
        assert config.hierarchy.max_clusters == 15

    def test_config_to_dict(self):
        """Test config serialization."""
        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "hierarchy": {
                    "enabled": True,
                    "grouping_strategy": "semantic",
                },
            }
        )

        d = config.to_dict()
        assert d["hierarchy"]["grouping_strategy"] == "semantic"


class TestHierarchyEnricherSemantic:
    """Tests for HierarchyEnricher with semantic grouping."""

    def test_enricher_init_semantic(self):
        """Test enricher initialization with semantic grouping."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "Summary"

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [1.0, 0.0, 0.0]

        config = HierarchyConfig(
            enabled=True,
            grouping_strategy="semantic",
            n_clusters=3,
        )

        enricher = HierarchyEnricher(
            config=config,
            chat_client=mock_chat,
            embedder=mock_embedder,
        )

        assert enricher._semantic_grouper is not None
        assert enricher._embedding_provider is not None

    def test_enricher_init_semantic_no_embedder_raises(self):
        """Test enricher raises error when semantic grouping without embedder."""
        mock_chat = MagicMock()

        config = HierarchyConfig(
            enabled=True,
            grouping_strategy="semantic",
        )

        # No embedder provided - should raise
        with pytest.raises(ValueError, match="Semantic grouping requires an embedder"):
            HierarchyEnricher(
                config=config,
                chat_client=mock_chat,
                embedder=None,
            )

    def test_enricher_semantic_grouping(self):
        """Test actual semantic grouping in enricher."""
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "Group summary"

        # Mock embedder returns different embeddings for different topics
        def mock_embed(text: str) -> list:
            if "Topic A" in text:
                return [1.0, 0.0, 0.0]
            else:
                return [0.0, 1.0, 0.0]

        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = mock_embed

        config = HierarchyConfig(
            enabled=True,
            grouping_strategy="semantic",
            n_clusters=2,
        )

        enricher = HierarchyEnricher(
            config=config,
            chat_client=mock_chat,
            embedder=mock_embedder,
        )

        chunks = [
            Chunk(id="c1", doc_id="d1", chunk_index=0, content="Topic A content 1", metadata={}),
            Chunk(id="c2", doc_id="d1", chunk_index=1, content="Topic A content 2", metadata={}),
            Chunk(id="c3", doc_id="d2", chunk_index=0, content="Topic B content 1", metadata={}),
            Chunk(id="c4", doc_id="d2", chunk_index=1, content="Topic B content 2", metadata={}),
        ]

        result = enricher.enrich(chunks)

        # Should have original chunks + corpus summary
        assert len(result) >= len(chunks)

        # Original chunks should have hierarchy metadata
        for chunk in chunks:
            assert "hierarchy_level" in chunk.metadata
            assert "hierarchy_group" in chunk.metadata
            assert chunk.metadata["hierarchy_group"].startswith("cluster_")


class TestEnrichmentPipelineSemantic:
    """Tests for EnrichmentPipeline with semantic grouping."""

    def test_pipeline_with_embedder(self, tmp_path):
        """Test pipeline initialization with embedder."""
        mock_chat = MagicMock()
        mock_embedder = MagicMock()

        config = EnrichmentConfig.from_dict(
            {
                "enabled": True,
                "hierarchy": {
                    "enabled": True,
                    "grouping_strategy": "semantic",
                },
            }
        )

        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=mock_chat,
            embedder=mock_embedder,
        )

        assert pipeline._hierarchy_enricher is not None
        assert pipeline._hierarchy_enricher._semantic_grouper is not None

    def test_pipeline_from_config_with_embedder(self, tmp_path):
        """Test pipeline.from_config with embedder."""
        mock_chat = MagicMock()
        mock_embedder = MagicMock()

        pipeline = EnrichmentPipeline.from_config(
            config={
                "enabled": True,
                "hierarchy": {
                    "enabled": True,
                    "grouping_strategy": "semantic",
                },
            },
            project_root=tmp_path,
            chat_client=mock_chat,
            embedder=mock_embedder,
        )

        assert pipeline._hierarchy_enricher is not None
        assert pipeline._hierarchy_enricher._semantic_grouper is not None
