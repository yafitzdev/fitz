# tests/test_clara_engine.py
"""
Tests for CLaRa Engine.

These tests verify the CLaRa engine implementation works correctly
and integrates properly with the Fitz platform.

Requires torch to be installed.
"""

from unittest.mock import MagicMock, patch

import pytest

# Core imports (torch-independent)
from fitz_ai.core import (
    Answer,
    Constraints,
    KnowledgeError,
    Provenance,
    Query,
)

# Skip entire module if torch not available
torch = pytest.importorskip("torch", reason="torch not installed")


class TestClaraConfig:
    """Test CLaRa configuration classes."""

    def test_default_config(self):
        """Test default configuration values."""
        from fitz_ai.engines.clara.config.schema import (
            ClaraConfig,
        )

        config = ClaraConfig()

        assert config.model.model_name_or_path == "apple/CLaRa-7B-Instruct/compression-16"
        assert config.model.variant == "instruct"
        assert config.model.load_in_4bit is True  # 4-bit enabled by default
        assert config.compression.compression_rate == 16
        assert config.retrieval.top_k == 5
        assert config.generation.max_new_tokens == 256

    def test_custom_config(self):
        """Test custom configuration."""
        from fitz_ai.engines.clara.config.schema import (
            ClaraCompressionConfig,
            ClaraConfig,
            ClaraModelConfig,
        )

        config = ClaraConfig(
            model=ClaraModelConfig(
                model_name_or_path="apple/CLaRa-7B-Instruct",
                variant="instruct",
                load_in_4bit=False,  # Override default
            ),
            compression=ClaraCompressionConfig(
                compression_rate=32,
            ),
        )

        assert config.model.model_name_or_path == "apple/CLaRa-7B-Instruct"
        assert config.model.variant == "instruct"
        assert config.model.load_in_4bit is False
        assert config.compression.compression_rate == 32

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading config from YAML file."""
        from fitz_ai.engines.clara.config.schema import load_clara_config

        config_content = """
clara:
  model:
    model_name_or_path: "test/model"
    variant: "base"
  compression:
    compression_rate: 64
  retrieval:
    top_k: 10
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)

        config = load_clara_config(str(config_file))

        assert config.model.model_name_or_path == "test/model"
        assert config.model.variant == "base"
        assert config.compression.compression_rate == 64
        assert config.retrieval.top_k == 10

    def test_load_config_missing_file(self):
        """Test loading config from missing file raises error."""
        from fitz_ai.engines.clara.config.schema import load_clara_config

        with pytest.raises(FileNotFoundError):
            load_clara_config("/nonexistent/path.yaml")


class TestClaraEngine:
    """Test CLaRa engine implementation."""

    @pytest.fixture
    def mock_engine(self):
        """Create a CLaRa engine with mocked model initialization."""
        from fitz_ai.engines.clara.config.schema import ClaraConfig
        from fitz_ai.engines.clara.engine import ClaraEngine

        config = ClaraConfig()

        # Mock _initialize_model to skip actual model loading
        with patch.object(ClaraEngine, "_initialize_model"):
            engine = ClaraEngine(config)

        # Create mock model with required methods
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()

        # Mock compress_documents to return proper tensor
        # Shape: [batch, num_mem_tokens, hidden_dim]
        def mock_compress(docs):
            batch = len(docs)
            return torch.randn(batch, 128, 4096)

        mock_model.compress_documents = mock_compress

        # Mock parameters() for device detection
        mock_param = torch.nn.Parameter(torch.zeros(1))
        mock_model.parameters = lambda: iter([mock_param])

        engine._model = mock_model

        return engine

    def test_engine_implements_protocol(self, mock_engine):
        """Test that ClaraEngine implements KnowledgeEngine protocol."""
        assert hasattr(mock_engine, "answer")
        assert callable(mock_engine.answer)

    def test_add_documents_compresses(self, mock_engine):
        """Test adding documents triggers pre-compression."""
        doc_ids = mock_engine.add_documents(
            [
                "Document 1",
                "Document 2",
                "Document 3",
            ]
        )

        assert len(doc_ids) == 3
        assert len(mock_engine._doc_texts) == 3
        assert mock_engine._doc_texts[0] == "Document 1"

        # Verify compressed representations were created
        assert mock_engine._compressed_docs is not None
        assert mock_engine._compressed_docs.shape[0] == 3  # 3 docs
        assert mock_engine._doc_embeddings is not None
        assert mock_engine._doc_embeddings.shape[0] == 3

    def test_add_documents_with_ids(self, mock_engine):
        """Test adding documents with custom IDs."""
        doc_ids = mock_engine.add_documents(["Doc 1", "Doc 2"], doc_ids=["custom_1", "custom_2"])

        assert doc_ids == ["custom_1", "custom_2"]
        assert len(mock_engine._doc_texts) == 2
        assert mock_engine._doc_ids == ["custom_1", "custom_2"]

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError during Query creation."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="   ")

    def test_answer_no_documents_raises_error(self, mock_engine):
        """Test that query without documents raises KnowledgeError."""
        assert mock_engine._doc_texts == []
        assert mock_engine._compressed_docs is None

        with pytest.raises(KnowledgeError, match="No documents"):
            mock_engine.answer(Query(text="What is X?"))

    def test_answer_success(self, mock_engine):
        """Test successful answer generation with latent retrieval."""
        # Add documents (this also creates compressed representations)
        mock_engine.add_documents(
            [
                "Document 1 content about quantum computing...",
                "Document 2 content about machine learning...",
            ]
        )

        # Mock model's generate methods
        # MagicMock auto-creates attributes, so we need to mock the compressed method too
        mock_engine._model.generate_from_compressed_documents_and_questions = MagicMock(
            return_value=["Quantum computing uses qubits."]
        )
        mock_engine._model.generate_from_text = MagicMock(
            return_value=["Quantum computing uses qubits."]
        )

        # Mock _retrieve_top_k to return known indices and similarities
        mock_engine._retrieve_top_k = MagicMock(return_value=([0, 1], torch.tensor([0.95, 0.85])))

        query = Query(text="What is quantum computing?")
        answer = mock_engine.answer(query)

        assert isinstance(answer, Answer)
        assert answer.text == "Quantum computing uses qubits."
        assert len(answer.provenance) == 2
        assert answer.metadata["engine"] == "clara"
        assert answer.metadata["retrieval_method"] == "cosine_similarity_latent"

    def test_answer_respects_constraints(self, mock_engine):
        """Test that answer respects query constraints."""
        # Add 10 documents
        mock_engine.add_documents([f"Content {i}" for i in range(10)])

        # Mock model methods (need both due to MagicMock auto-attribute creation)
        mock_engine._model.generate_from_compressed_documents_and_questions = MagicMock(
            return_value=["Answer"]
        )
        mock_engine._model.generate_from_text = MagicMock(return_value=["Answer"])

        # Mock _retrieve_top_k to return 3 results (respecting constraint)
        mock_engine._retrieve_top_k = MagicMock(
            return_value=([0, 1, 2], torch.tensor([0.9, 0.8, 0.7]))
        )

        query = Query(text="Question?", constraints=Constraints(max_sources=3))
        answer = mock_engine.answer(query)

        # Verify only 3 sources in provenance (respecting constraint)
        assert len(answer.provenance) == 3
        assert answer.metadata["num_docs_retrieved"] == 3
        # Verify _retrieve_top_k was called with correct top_k
        mock_engine._retrieve_top_k.assert_called_once_with("Question?", 3)

    def test_retrieval_uses_cosine_similarity(self, mock_engine):
        """Test that retrieval returns semantically similar docs."""
        # Add documents with distinct embeddings
        mock_engine.add_documents(["Doc A", "Doc B", "Doc C"])

        # Manually set embeddings to test retrieval logic
        # Make Doc B most similar to query
        mock_engine._doc_embeddings = torch.tensor(
            [
                [0.1, 0.0, 0.0],  # Doc A
                [0.9, 0.1, 0.0],  # Doc B - most similar to query
                [0.2, 0.1, 0.0],  # Doc C
            ]
        )

        # Mock encode_query to return embedding similar to Doc B
        mock_engine._model.encode_query = MagicMock(return_value=torch.tensor([[0.9, 0.1, 0.0]]))

        indices, similarities = mock_engine._retrieve_top_k("test query", top_k=2)

        # Doc B (index 1) should be first due to highest similarity
        assert indices[0] == 1

    def test_get_knowledge_stats(self, mock_engine):
        """Test knowledge base statistics."""
        mock_engine.add_documents(["Doc 1", "Doc 2"])

        stats = mock_engine.get_knowledge_stats()

        assert stats["num_documents"] == 2
        assert "compression_rate" in stats
        assert "model_variant" in stats
        assert "quantization" in stats
        assert stats["quantization"] == "4-bit"  # Default
        assert "compressed_shape" in stats
        assert "compressed_memory_mb" in stats

    def test_clear_knowledge_base(self, mock_engine):
        """Test clearing knowledge base."""
        mock_engine.add_documents(["Doc 1", "Doc 2"])
        assert mock_engine._compressed_docs is not None

        mock_engine.clear_knowledge_base()

        assert len(mock_engine._doc_texts) == 0
        assert len(mock_engine._doc_ids) == 0
        assert mock_engine._compressed_docs is None
        assert mock_engine._doc_embeddings is None


class TestClaraRuntime:
    """Test CLaRa runtime convenience functions."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear engine cache before each test."""
        from fitz_ai.engines.clara.runtime import clear_engine_cache

        clear_engine_cache()
        yield
        clear_engine_cache()

    def test_create_clara_engine(self):
        """Test create_clara_engine factory."""
        from fitz_ai.engines.clara.engine import ClaraEngine
        from fitz_ai.engines.clara.runtime import create_clara_engine

        # Mock _initialize_model to skip actual model loading
        with patch.object(ClaraEngine, "_initialize_model"):
            engine = create_clara_engine()

        assert engine is not None
        assert isinstance(engine, ClaraEngine)


class TestClaraRegistration:
    """Test CLaRa engine registration with global registry."""

    def test_clara_registered(self):
        """Test that CLaRa is registered in the engine registry."""
        # Import clara module to trigger registration
        import fitz_ai.engines.clara  # noqa
        from fitz_ai.runtime import get_engine_registry

        registry = get_engine_registry()
        engines = registry.list()

        assert "clara" in engines

    def test_clara_capabilities(self):
        """Test that CLaRa has correct capabilities."""
        import fitz_ai.engines.clara  # noqa
        from fitz_ai.runtime import get_engine_registry

        registry = get_engine_registry()
        caps = registry.get_capabilities("clara")

        assert caps.supports_collections is False
        assert caps.requires_documents_at_query is False  # Has persistent storage
        assert caps.supports_chat is False
        assert caps.requires_api_key is False


class TestClaraIntegration:
    """Integration tests for CLaRa with the Fitz platform."""

    def test_clara_with_universal_run(self):
        """Test using CLaRa via the universal run() function."""
        # This test would require the full setup
        # Skipping for now as it needs model loading
        pass

    def test_clara_answer_format_matches_fitz_rag(self):
        """Test that CLaRa answers have the same format as FitzRAG."""
        from fitz_ai.core import Answer

        # Both engines should return Answer objects
        answer = Answer(
            text="Test answer",
            provenance=[Provenance(source_id="doc_1")],
            metadata={"engine": "clara"},
        )

        assert hasattr(answer, "text")
        assert hasattr(answer, "provenance")
        assert hasattr(answer, "metadata")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
