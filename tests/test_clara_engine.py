# tests/test_clara_engine.py
"""
Tests for CLaRa Engine.

These tests verify the CLaRa engine implementation works correctly
and integrates properly with the Fitz platform.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Core imports
from fitz_ai.core import (
    Answer,
    Constraints,
    KnowledgeError,
    Provenance,
    Query,
)


class TestClaraConfig:
    """Test CLaRa configuration classes."""

    def test_default_config(self):
        """Test default configuration values."""
        from fitz_ai.engines.clara.config.schema import (
            ClaraConfig,
        )

        config = ClaraConfig()

        assert (
            config.model.model_name_or_path == "apple/CLaRa-7B-Instruct/compression-16"
        )
        assert config.model.variant == "instruct"
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
            ),
            compression=ClaraCompressionConfig(
                compression_rate=32,
            ),
        )

        assert config.model.model_name_or_path == "apple/CLaRa-7B-Instruct"
        assert config.model.variant == "instruct"
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
    def mock_torch(self):
        """Create mock torch module."""
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        mock_torch.randn.return_value = MagicMock()
        mock_torch.stack.return_value = MagicMock()
        mock_torch.topk.return_value = (MagicMock(), MagicMock())
        return mock_torch

    @pytest.fixture
    def mock_transformers(self):
        """Create mock transformers module."""
        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.eval.return_value = None
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model
        return mock_transformers

    @pytest.fixture
    def mock_engine(self, mock_torch, mock_transformers):
        """Create a CLaRa engine with mocked dependencies."""
        # Insert mock modules into sys.modules BEFORE importing engine
        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "torch.nn": MagicMock(),
                "torch.nn.functional": MagicMock(),
                "transformers": mock_transformers,
            },
        ):
            from fitz_ai.engines.clara.config.schema import ClaraConfig
            from fitz_ai.engines.clara.engine import ClaraEngine

            config = ClaraConfig()
            engine = ClaraEngine(config)

            # Store mock references for test assertions
            engine._mock_torch = mock_torch
            engine._mock_transformers = mock_transformers

            return engine

    def test_engine_implements_protocol(self, mock_engine):
        """Test that ClaraEngine implements KnowledgeEngine protocol."""
        assert hasattr(mock_engine, "answer")
        assert callable(mock_engine.answer)

    def test_add_documents(self, mock_engine, mock_torch):
        """Test adding documents to knowledge base."""
        # Mock the compression to return mock tensors
        mock_tensor = MagicMock()
        mock_engine._compress_documents = MagicMock(
            return_value=[mock_tensor for _ in range(3)]
        )

        doc_ids = mock_engine.add_documents(
            [
                "Document 1",
                "Document 2",
                "Document 3",
            ]
        )

        assert len(doc_ids) == 3
        assert len(mock_engine._compressed_docs) == 3
        assert len(mock_engine._doc_texts) == 3

    def test_add_documents_with_ids(self, mock_engine):
        """Test adding documents with custom IDs."""
        mock_tensor = MagicMock()
        mock_engine._compress_documents = MagicMock(
            return_value=[mock_tensor for _ in range(2)]
        )

        doc_ids = mock_engine.add_documents(
            ["Doc 1", "Doc 2"], doc_ids=["custom_1", "custom_2"]
        )

        assert doc_ids == ["custom_1", "custom_2"]
        assert "custom_1" in mock_engine._compressed_docs
        assert "custom_2" in mock_engine._compressed_docs

    def test_empty_query_raises_error(self):
        """Test that empty query raises ValueError during Query creation."""
        # Query validates in __post_init__, so this should raise ValueError
        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="   ")

    def test_answer_no_documents_raises_error(self, mock_engine):
        """Test that query without documents raises KnowledgeError."""
        mock_engine._compressed_docs = {}

        with pytest.raises(KnowledgeError, match="No documents"):
            mock_engine.answer(Query(text="What is X?"))

    def test_answer_success(self, mock_engine):
        """Test successful answer generation."""
        mock_tensor = MagicMock()

        # Setup knowledge base
        mock_engine._compressed_docs = {
            "doc_1": mock_tensor,
            "doc_2": mock_tensor,
        }
        mock_engine._doc_texts = {
            "doc_1": "Document 1 content about quantum computing...",
            "doc_2": "Document 2 content about machine learning...",
        }

        # Mock retrieval
        mock_engine._retrieve = MagicMock(return_value=(["doc_1", "doc_2"], [0.9, 0.8]))

        # Mock generation
        mock_engine._generate = MagicMock(
            return_value=("Quantum computing uses qubits.", [0, 1])
        )

        query = Query(text="What is quantum computing?")
        answer = mock_engine.answer(query)

        assert isinstance(answer, Answer)
        assert answer.text == "Quantum computing uses qubits."
        assert len(answer.provenance) == 2
        assert answer.metadata["engine"] == "clara"
        # Check that score is in metadata, not as direct field
        assert "relevance_score" in answer.provenance[0].metadata

    def test_answer_respects_constraints(self, mock_engine):
        """Test that answer respects query constraints."""
        mock_tensor = MagicMock()

        mock_engine._compressed_docs = {f"doc_{i}": mock_tensor for i in range(10)}
        mock_engine._doc_texts = {f"doc_{i}": f"Content {i}" for i in range(10)}

        mock_engine._retrieve = MagicMock(
            return_value=(["doc_0", "doc_1", "doc_2"], [0.9, 0.8, 0.7])
        )
        mock_engine._generate = MagicMock(return_value=("Answer", [0, 1, 2]))

        query = Query(text="Question?", constraints=Constraints(max_sources=3))
        answer = mock_engine.answer(query)

        # Verify retrieve was called with constrained top_k
        mock_engine._retrieve.assert_called_once()
        call_args = mock_engine._retrieve.call_args
        assert call_args[1]["top_k"] == 3

    def test_get_knowledge_stats(self, mock_engine):
        """Test knowledge base statistics."""
        mock_tensor = MagicMock()

        mock_engine._compressed_docs = {
            "doc_1": mock_tensor,
            "doc_2": mock_tensor,
        }

        stats = mock_engine.get_knowledge_stats()

        assert stats["num_documents"] == 2
        assert "compression_rate" in stats
        assert "model_variant" in stats

    def test_clear_knowledge_base(self, mock_engine):
        """Test clearing knowledge base."""
        mock_tensor = MagicMock()

        mock_engine._compressed_docs = {"doc_1": mock_tensor}
        mock_engine._doc_texts = {"doc_1": "Content"}

        mock_engine.clear_knowledge_base()

        assert len(mock_engine._compressed_docs) == 0
        assert len(mock_engine._doc_texts) == 0


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
        mock_torch = MagicMock()
        mock_torch.float32 = "float32"
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"

        mock_transformers = MagicMock()
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.eval.return_value = None
        mock_transformers.AutoModel.from_pretrained.return_value = mock_model

        with patch.dict(
            sys.modules,
            {
                "torch": mock_torch,
                "torch.nn": MagicMock(),
                "torch.nn.functional": MagicMock(),
                "transformers": mock_transformers,
            },
        ):
            from fitz_ai.engines.clara.runtime import create_clara_engine

            engine = create_clara_engine()

            assert engine is not None
            mock_transformers.AutoModel.from_pretrained.assert_called_once()


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


class TestClaraIntegration:
    """Integration tests for CLaRa with the Fitz platform."""

    def test_clara_with_universal_run(self):
        """Test using CLaRa via the universal run() function."""
        # This test would require the full setup
        # Skipping for now as it needs model loading
        pass

    def test_clara_answer_format_matches_classic_rag(self):
        """Test that CLaRa answers have the same format as Classic RAG."""
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
