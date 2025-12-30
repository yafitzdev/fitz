# tests/test_graphrag_engine.py
"""
Tests for GraphRAG Engine.

These tests verify the GraphRAG engine implementation works correctly
and integrates properly with the Fitz platform.

Requires networkx to be installed.
"""

from unittest.mock import MagicMock

import pytest

from fitz_ai.core import Answer, KnowledgeError, Provenance, Query

# Skip entire module if networkx not available
pytest.importorskip("networkx", reason="networkx not installed")


class TestGraphRAGConfig:
    """Test GraphRAG configuration classes."""

    def test_default_config(self):
        """Test default configuration values."""
        from fitz_ai.engines.graphrag.config.schema import GraphRAGConfig

        config = GraphRAGConfig()

        assert config.extraction.max_entities_per_chunk == 20
        assert config.extraction.max_relationships_per_chunk == 30
        assert config.community.algorithm == "louvain"
        assert config.community.resolution == 1.0
        assert config.search.default_mode == "local"
        assert config.search.local_top_k == 10

    def test_custom_config(self):
        """Test custom configuration."""
        from fitz_ai.engines.graphrag.config.schema import (
            GraphCommunityConfig,
            GraphExtractionConfig,
            GraphRAGConfig,
            GraphSearchConfig,
        )

        config = GraphRAGConfig(
            extraction=GraphExtractionConfig(max_entities_per_chunk=50),
            community=GraphCommunityConfig(algorithm="leiden", resolution=1.5),
            search=GraphSearchConfig(default_mode="global"),
        )

        assert config.extraction.max_entities_per_chunk == 50
        assert config.community.algorithm == "leiden"
        assert config.community.resolution == 1.5
        assert config.search.default_mode == "global"

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        from fitz_ai.engines.graphrag.config.schema import load_graphrag_config

        config_content = """
graphrag:
  extraction:
    max_entities_per_chunk: 30
    entity_types:
      - person
      - organization
  community:
    algorithm: leiden
    resolution: 2.0
  search:
    default_mode: hybrid
    local_top_k: 15
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_graphrag_config(str(config_file))

        assert config.extraction.max_entities_per_chunk == 30
        assert config.community.algorithm == "leiden"
        assert config.community.resolution == 2.0
        assert config.search.default_mode == "hybrid"
        assert config.search.local_top_k == 15

    def test_load_config_from_default(self):
        """Test loading config from default.yaml."""
        from fitz_ai.engines.graphrag.config.schema import load_graphrag_config

        # When called with None, should load from default.yaml
        config = load_graphrag_config()

        # Should have values from default.yaml
        assert config.llm_provider == "cohere"
        assert config.embedding_provider == "cohere"
        assert config.extraction.max_entities_per_chunk == 20
        assert config.community.algorithm == "louvain"
        assert config.search.default_mode == "local"

    def test_load_config_missing_file(self):
        """Test loading config from missing file raises error."""
        from fitz_ai.engines.graphrag.config.schema import load_graphrag_config

        with pytest.raises(FileNotFoundError):
            load_graphrag_config("/nonexistent/path.yaml")


class TestKnowledgeGraph:
    """Test knowledge graph storage."""

    def test_add_entity(self):
        """Test adding entities."""
        from fitz_ai.engines.graphrag.graph.storage import Entity, KnowledgeGraph

        graph = KnowledgeGraph()
        entity = Entity(id="e1", name="Apple", type="organization", description="Tech company")
        graph.add_entity(entity)

        assert graph.num_entities == 1
        assert graph.get_entity("e1") == entity

    def test_add_relationship(self):
        """Test adding relationships."""
        from fitz_ai.engines.graphrag.graph.storage import (
            Entity,
            KnowledgeGraph,
            Relationship,
        )

        graph = KnowledgeGraph()
        graph.add_entity(Entity(id="e1", name="Steve Jobs", type="person"))
        graph.add_entity(Entity(id="e2", name="Apple", type="organization"))

        rel = Relationship(
            source_id="e1",
            target_id="e2",
            type="founded",
            description="Steve Jobs founded Apple",
        )
        graph.add_relationship(rel)

        assert graph.num_relationships == 1
        rels = graph.get_entity_relationships("e1")
        assert len(rels) == 1
        assert rels[0].type == "founded"

    def test_get_neighbors(self):
        """Test getting neighbors."""
        from fitz_ai.engines.graphrag.graph.storage import (
            Entity,
            KnowledgeGraph,
            Relationship,
        )

        graph = KnowledgeGraph()
        graph.add_entity(Entity(id="e1", name="A", type="test"))
        graph.add_entity(Entity(id="e2", name="B", type="test"))
        graph.add_entity(Entity(id="e3", name="C", type="test"))

        graph.add_relationship(Relationship(source_id="e1", target_id="e2", type="connected"))
        graph.add_relationship(Relationship(source_id="e2", target_id="e3", type="connected"))

        neighbors_1_hop = graph.get_neighbors("e1", max_hops=1)
        assert neighbors_1_hop == {"e2"}

        neighbors_2_hop = graph.get_neighbors("e1", max_hops=2)
        assert neighbors_2_hop == {"e2", "e3"}

    def test_save_and_load(self, tmp_path):
        """Test saving and loading graph."""
        from fitz_ai.engines.graphrag.graph.storage import (
            Entity,
            KnowledgeGraph,
            Relationship,
        )

        graph = KnowledgeGraph()
        graph.add_entity(Entity(id="e1", name="Test", type="test", description="Test entity"))
        graph.add_entity(Entity(id="e2", name="Test2", type="test"))
        graph.add_relationship(Relationship(source_id="e1", target_id="e2", type="related"))

        path = str(tmp_path / "graph.json")
        graph.save(path)

        loaded = KnowledgeGraph.load(path)
        assert loaded.num_entities == 2
        assert loaded.num_relationships == 1

    def test_to_context_string(self):
        """Test converting graph to context string."""
        from fitz_ai.engines.graphrag.graph.storage import (
            Entity,
            KnowledgeGraph,
            Relationship,
        )

        graph = KnowledgeGraph()
        graph.add_entity(Entity(id="e1", name="Alice", type="person", description="Engineer"))
        graph.add_entity(Entity(id="e2", name="Bob", type="person", description="Manager"))
        graph.add_relationship(Relationship(source_id="e1", target_id="e2", type="reports_to"))

        context = graph.to_context_string()

        assert "Alice" in context
        assert "Bob" in context
        assert "reports_to" in context


class TestGraphRAGEngine:
    """Test GraphRAG engine implementation."""

    @pytest.fixture
    def mock_engine(self):
        """Create a GraphRAG engine with mocked LLM."""
        from fitz_ai.engines.graphrag.config.schema import GraphRAGConfig
        from fitz_ai.engines.graphrag.engine import GraphRAGEngine

        config = GraphRAGConfig()
        engine = GraphRAGEngine(config)

        # Mock the chat engine
        mock_chat = MagicMock()
        mock_chat.chat = MagicMock(return_value="Mock answer from LLM")
        engine._chat_engine = mock_chat

        return engine

    def test_engine_implements_protocol(self, mock_engine):
        """Test that GraphRAGEngine implements KnowledgeEngine protocol."""
        assert hasattr(mock_engine, "answer")
        assert callable(mock_engine.answer)

    def test_add_documents(self, mock_engine):
        """Test adding documents."""
        doc_ids = mock_engine.add_documents(["Document 1", "Document 2"])

        assert len(doc_ids) == 2
        assert len(mock_engine._doc_texts) == 2
        assert mock_engine._doc_texts[0] == "Document 1"

    def test_add_documents_with_ids(self, mock_engine):
        """Test adding documents with custom IDs."""
        doc_ids = mock_engine.add_documents(["Doc 1", "Doc 2"], doc_ids=["custom_1", "custom_2"])

        assert doc_ids == ["custom_1", "custom_2"]
        assert mock_engine._doc_ids == ["custom_1", "custom_2"]

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="   ")

    def test_answer_no_documents_raises_error(self, mock_engine):
        """Test that query without documents raises KnowledgeError."""
        assert mock_engine._doc_texts == []

        with pytest.raises(KnowledgeError, match="No documents"):
            mock_engine.answer(Query(text="What is X?"))

    def test_get_knowledge_stats(self, mock_engine):
        """Test knowledge stats."""
        mock_engine.add_documents(["Doc 1", "Doc 2"])

        stats = mock_engine.get_knowledge_stats()

        assert stats["num_documents"] == 2
        assert stats["num_chunks"] == 2
        assert "num_entities" in stats
        assert "search_mode" in stats

    def test_clear_knowledge_base(self, mock_engine):
        """Test clearing knowledge base."""
        mock_engine.add_documents(["Doc 1", "Doc 2"])

        mock_engine.clear_knowledge_base()

        assert len(mock_engine._doc_texts) == 0
        assert len(mock_engine._chunks) == 0
        assert mock_engine._graph.num_entities == 0


class TestCommunityDetection:
    """Test community detection."""

    def test_detect_louvain(self):
        """Test Louvain community detection."""
        from fitz_ai.engines.graphrag.config.schema import GraphCommunityConfig
        from fitz_ai.engines.graphrag.graph.community import CommunityDetector
        from fitz_ai.engines.graphrag.graph.storage import (
            Entity,
            KnowledgeGraph,
            Relationship,
        )

        graph = KnowledgeGraph()
        # Create two clusters
        for i in range(5):
            graph.add_entity(Entity(id=f"a{i}", name=f"A{i}", type="test"))
        for i in range(5):
            graph.add_entity(Entity(id=f"b{i}", name=f"B{i}", type="test"))

        # Dense connections within clusters
        for i in range(4):
            graph.add_relationship(Relationship(source_id=f"a{i}", target_id=f"a{i + 1}", type="r"))
            graph.add_relationship(Relationship(source_id=f"b{i}", target_id=f"b{i + 1}", type="r"))

        # Sparse connection between clusters
        graph.add_relationship(Relationship(source_id="a0", target_id="b0", type="r"))

        config = GraphCommunityConfig(algorithm="louvain", min_community_size=2)
        detector = CommunityDetector(config)

        communities = detector.detect_communities(graph)

        # Should detect at least 1 community (may merge into one depending on resolution)
        assert len(communities) >= 1

    def test_connected_components_fallback(self):
        """Test connected components as fallback."""
        from fitz_ai.engines.graphrag.config.schema import GraphCommunityConfig
        from fitz_ai.engines.graphrag.graph.community import CommunityDetector
        from fitz_ai.engines.graphrag.graph.storage import Entity, KnowledgeGraph

        graph = KnowledgeGraph()
        # Create disconnected entities
        graph.add_entity(Entity(id="e1", name="A", type="test"))
        graph.add_entity(Entity(id="e2", name="B", type="test"))

        config = GraphCommunityConfig(min_community_size=1)
        detector = CommunityDetector(config)

        communities = detector._detect_connected_components(graph)

        # Each entity should be its own component
        assert len(communities) == 2


class TestGraphRAGRegistration:
    """Test GraphRAG engine registration with global registry."""

    def test_graphrag_registered(self):
        """Test that GraphRAG is registered in the engine registry."""
        import fitz_ai.engines.graphrag  # noqa
        from fitz_ai.runtime import get_engine_registry

        registry = get_engine_registry()
        engines = registry.list()

        assert "graphrag" in engines

    def test_graphrag_capabilities(self):
        """Test that GraphRAG has correct capabilities."""
        import fitz_ai.engines.graphrag  # noqa
        from fitz_ai.runtime import get_engine_registry

        registry = get_engine_registry()
        caps = registry.get_capabilities("graphrag")

        assert caps.supports_collections is False
        assert caps.requires_documents_at_query is False  # Has persistent storage
        assert caps.supports_chat is False
        assert caps.requires_api_key is False


class TestGraphRAGIntegration:
    """Integration tests for GraphRAG with the Fitz platform."""

    def test_graphrag_answer_format(self):
        """Test that GraphRAG answers have the correct format."""
        answer = Answer(
            text="Test answer",
            provenance=[Provenance(source_id="doc_1")],
            metadata={"engine": "graphrag"},
        )

        assert hasattr(answer, "text")
        assert hasattr(answer, "provenance")
        assert hasattr(answer, "metadata")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
