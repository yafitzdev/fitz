# tests/test_cli_init.py
"""
Tests for the init command.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestInitCommand:
    """Tests for fitz init command."""

    def test_init_shows_help(self):
        """Test that init --help works."""
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "init" in result.output.lower()
        assert "interactive" in result.output.lower() or "wizard" in result.output.lower()

    def test_init_show_config_flag(self):
        """Test that --show flag previews config without saving."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.qdrant.available = False
        mock_system.faiss.available = True
        mock_system.api_keys = {
            "cohere": MagicMock(available=True),
            "openai": MagicMock(available=False),
            "anthropic": MagicMock(available=False),
        }

        with (
            patch("fitz_ai.cli.commands.init.detect_system", return_value=mock_system),
            patch(
                "fitz_ai.cli.commands.init.available_llm_plugins",
                side_effect=lambda t: ["cohere"] if t in ["chat", "embedding"] else [],
            ),
            patch(
                "fitz_ai.cli.commands.init.available_vector_db_plugins",
                return_value=["local_faiss"],
            ),
            patch(
                "fitz_ai.cli.commands.init.available_retrieval_plugins",
                return_value=["dense"],
            ),
            patch(
                "fitz_ai.cli.commands.init.available_chunking_plugins",
                return_value=["simple"],
            ),
            patch("fitz_ai.cli.commands.init._load_default_config", return_value={}),
        ):
            result = runner.invoke(app, ["init", "-y", "--show"])

        # Should not save (exit before save)
        assert result.exit_code == 0


class TestInitHelpers:
    """Tests for init helper functions."""

    def test_filter_available_plugins_ollama(self):
        """Test _filter_available_plugins filters Ollama plugins."""
        from fitz_ai.cli.commands.init import _filter_available_plugins

        mock_system = MagicMock()
        mock_system.ollama.available = True

        plugins = ["cohere", "local_ollama", "openai"]
        result = _filter_available_plugins(plugins, "chat", mock_system)

        assert "local_ollama" in result

    def test_filter_available_plugins_ollama_unavailable(self):
        """Test _filter_available_plugins excludes Ollama when unavailable."""
        from fitz_ai.cli.commands.init import _filter_available_plugins

        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.api_keys = {}

        plugins = ["local_ollama"]
        result = _filter_available_plugins(plugins, "chat", mock_system)

        assert "local_ollama" not in result

    def test_filter_available_plugins_faiss(self):
        """Test _filter_available_plugins includes FAISS when available."""
        from fitz_ai.cli.commands.init import _filter_available_plugins

        mock_system = MagicMock()
        mock_system.faiss.available = True

        plugins = ["local_faiss", "qdrant"]
        result = _filter_available_plugins(plugins, "vector_db", mock_system)

        assert "local_faiss" in result

    def test_filter_available_plugins_api_keys(self):
        """Test _filter_available_plugins checks API keys."""
        from fitz_ai.cli.commands.init import _filter_available_plugins

        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.qdrant.available = False
        mock_system.faiss.available = False
        mock_system.api_keys = {
            "cohere": MagicMock(available=True),
            "openai": MagicMock(available=False),
        }

        plugins = ["cohere", "openai"]
        result = _filter_available_plugins(plugins, "chat", mock_system)

        assert "cohere" in result
        assert "openai" not in result

    def test_get_default_model_chat(self):
        """Test _get_default_model returns correct chat models for smart/fast tiers."""
        from fitz_ai.cli.commands.init import _get_default_model

        # Smart tier (default)
        assert _get_default_model("chat", "cohere", "smart") == "command-a-03-2025"
        assert _get_default_model("chat", "openai", "smart") == "gpt-4o"
        assert "llama" in _get_default_model("chat", "local_ollama", "smart")

        # Fast tier
        assert _get_default_model("chat", "cohere", "fast") == "command-r7b-12-2024"
        assert _get_default_model("chat", "openai", "fast") == "gpt-4o-mini"
        assert "llama" in _get_default_model("chat", "local_ollama", "fast")

    def test_get_default_model_embedding(self):
        """Test _get_default_model returns correct embedding model."""
        from fitz_ai.cli.commands.init import _get_default_model

        assert _get_default_model("embedding", "cohere") == "embed-english-v3.0"
        assert _get_default_model("embedding", "openai") == "text-embedding-3-small"

    def test_get_default_model_unknown(self):
        """Test _get_default_model returns empty for unknown plugin."""
        from fitz_ai.cli.commands.init import _get_default_model

        assert _get_default_model("chat", "unknown_plugin") == ""


class TestGenerateConfig:
    """Tests for config generation functions."""

    def test_generate_global_config(self):
        """Test _generate_global_config produces valid YAML."""
        import yaml

        from fitz_ai.cli.commands.init import _generate_global_config

        config_str = _generate_global_config("classic_rag")
        config = yaml.safe_load(config_str)

        assert config["default_engine"] == "classic_rag"

    def test_generate_classic_rag_config_basic(self):
        """Test _generate_classic_rag_config produces valid YAML."""
        import yaml

        from fitz_ai.cli.commands.init import _generate_classic_rag_config

        config_str = _generate_classic_rag_config(
            chat="cohere",
            chat_model_smart="command-a-03-2025",
            chat_model_fast="command-r7b-12-2024",
            embedding="cohere",
            embedding_model="embed-english-v3.0",
            rerank=None,
            rerank_model="",
            vector_db="local_faiss",
            retrieval="dense",
            qdrant_host="localhost",
            qdrant_port=6333,
            chunker="simple",
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Should be valid YAML
        config = yaml.safe_load(config_str)

        assert config["chat"]["plugin_name"] == "cohere"
        assert config["chat"]["kwargs"]["models"]["smart"] == "command-a-03-2025"
        assert config["chat"]["kwargs"]["models"]["fast"] == "command-r7b-12-2024"
        assert config["embedding"]["plugin_name"] == "cohere"
        assert config["vector_db"]["plugin_name"] == "local_faiss"
        assert config["retrieval"]["plugin_name"] == "dense"

    def test_generate_classic_rag_config_with_rerank(self):
        """Test _generate_classic_rag_config includes rerank when provided."""
        import yaml

        from fitz_ai.cli.commands.init import _generate_classic_rag_config

        config_str = _generate_classic_rag_config(
            chat="cohere",
            chat_model_smart="command-a-03-2025",
            chat_model_fast="command-r7b-12-2024",
            embedding="cohere",
            embedding_model="embed-english-v3.0",
            rerank="cohere",
            rerank_model="rerank-v3.5",
            vector_db="local_faiss",
            retrieval="dense",
            qdrant_host="localhost",
            qdrant_port=6333,
            chunker="simple",
            chunk_size=1000,
            chunk_overlap=200,
        )

        config = yaml.safe_load(config_str)

        assert config["rerank"]["enabled"] is True
        assert config["rerank"]["plugin_name"] == "cohere"

    def test_generate_classic_rag_config_without_rerank(self):
        """Test _generate_classic_rag_config disables rerank when not provided."""
        import yaml

        from fitz_ai.cli.commands.init import _generate_classic_rag_config

        config_str = _generate_classic_rag_config(
            chat="cohere",
            chat_model_smart="",
            chat_model_fast="",
            embedding="cohere",
            embedding_model="",
            rerank=None,
            rerank_model="",
            vector_db="local_faiss",
            retrieval="dense",
            qdrant_host="localhost",
            qdrant_port=6333,
            chunker="simple",
            chunk_size=1000,
            chunk_overlap=0,
        )

        config = yaml.safe_load(config_str)

        assert config["rerank"]["enabled"] is False

    def test_generate_classic_rag_config_qdrant(self):
        """Test _generate_classic_rag_config includes Qdrant settings."""
        import yaml

        from fitz_ai.cli.commands.init import _generate_classic_rag_config

        config_str = _generate_classic_rag_config(
            chat="cohere",
            chat_model_smart="",
            chat_model_fast="",
            embedding="cohere",
            embedding_model="",
            rerank=None,
            rerank_model="",
            vector_db="qdrant",
            retrieval="dense",
            qdrant_host="192.168.1.100",
            qdrant_port=6334,
            chunker="simple",
            chunk_size=1000,
            chunk_overlap=0,
        )

        config = yaml.safe_load(config_str)

        assert config["vector_db"]["plugin_name"] == "qdrant"
        assert config["vector_db"]["kwargs"]["host"] == "192.168.1.100"
        assert config["vector_db"]["kwargs"]["port"] == 6334

    def test_generate_classic_rag_config_chunking(self):
        """Test _generate_classic_rag_config includes chunking settings."""
        import yaml

        from fitz_ai.cli.commands.init import _generate_classic_rag_config

        config_str = _generate_classic_rag_config(
            chat="cohere",
            chat_model_smart="",
            chat_model_fast="",
            embedding="cohere",
            embedding_model="",
            rerank=None,
            rerank_model="",
            vector_db="local_faiss",
            retrieval="dense",
            qdrant_host="localhost",
            qdrant_port=6333,
            chunker="recursive",
            chunk_size=500,
            chunk_overlap=100,
        )

        config = yaml.safe_load(config_str)

        assert config["chunking"]["default"]["plugin_name"] == "recursive"
        assert config["chunking"]["default"]["kwargs"]["chunk_size"] == 500
        assert config["chunking"]["default"]["kwargs"]["chunk_overlap"] == 100

    def test_copy_engine_default_config_graphrag(self):
        """Test _copy_engine_default_config returns graphrag default config."""
        import yaml

        from fitz_ai.cli.commands.init import _copy_engine_default_config
        from fitz_ai.runtime import get_engine_registry

        # Ensure graphrag is registered
        import fitz_ai.engines.graphrag  # noqa: F401

        registry = get_engine_registry()
        config_str = _copy_engine_default_config("graphrag", registry)

        assert config_str is not None
        config = yaml.safe_load(config_str)

        assert "graphrag" in config
        assert "extraction" in config["graphrag"]
        assert "community" in config["graphrag"]
        assert "search" in config["graphrag"]

    def test_copy_engine_default_config_clara(self):
        """Test _copy_engine_default_config returns clara default config."""
        import yaml

        from fitz_ai.cli.commands.init import _copy_engine_default_config
        from fitz_ai.runtime import get_engine_registry

        # Ensure clara is registered
        import fitz_ai.engines.clara  # noqa: F401

        registry = get_engine_registry()
        config_str = _copy_engine_default_config("clara", registry)

        assert config_str is not None
        config = yaml.safe_load(config_str)

        assert "clara" in config
        assert "model" in config["clara"]
        assert "compression" in config["clara"]
        assert "retrieval" in config["clara"]
        assert "generation" in config["clara"]


class TestInitValidation:
    """Tests for init validation logic."""

    def test_init_fails_without_chat_plugins(self):
        """Test init fails when no chat plugins available (classic_rag)."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.qdrant.available = False
        mock_system.faiss.available = True
        mock_system.api_keys = {}

        with (
            patch("fitz_ai.cli.commands.init.detect_system", return_value=mock_system),
            patch("fitz_ai.cli.commands.init.get_default_engine", return_value="classic_rag"),
            patch("fitz_ai.cli.commands.init.available_llm_plugins", return_value=[]),
            patch(
                "fitz_ai.cli.commands.init.available_vector_db_plugins",
                return_value=["local_faiss"],
            ),
            patch(
                "fitz_ai.cli.commands.init.available_retrieval_plugins",
                return_value=["dense"],
            ),
            patch(
                "fitz_ai.cli.commands.init.available_chunking_plugins",
                return_value=["simple"],
            ),
            patch("fitz_ai.cli.commands.init._load_default_config", return_value={}),
        ):
            result = runner.invoke(app, ["init", "-y"])

        assert result.exit_code != 0
        assert "chat" in result.output.lower() or "api key" in result.output.lower()

    def test_init_fails_without_vector_db(self):
        """Test init fails when no vector DB available (classic_rag)."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.qdrant.available = False
        mock_system.faiss.available = False
        mock_system.api_keys = {"cohere": MagicMock(available=True)}

        with (
            patch("fitz_ai.cli.commands.init.detect_system", return_value=mock_system),
            patch("fitz_ai.cli.commands.init.get_default_engine", return_value="classic_rag"),
            patch(
                "fitz_ai.cli.commands.init.available_llm_plugins",
                return_value=["cohere"],
            ),
            patch("fitz_ai.cli.commands.init.available_vector_db_plugins", return_value=[]),
            patch(
                "fitz_ai.cli.commands.init.available_retrieval_plugins",
                return_value=["dense"],
            ),
            patch(
                "fitz_ai.cli.commands.init.available_chunking_plugins",
                return_value=["simple"],
            ),
            patch("fitz_ai.cli.commands.init._load_default_config", return_value={}),
        ):
            result = runner.invoke(app, ["init", "-y"])

        assert result.exit_code != 0
        assert "vector" in result.output.lower()
