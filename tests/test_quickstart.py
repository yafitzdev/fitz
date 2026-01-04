# tests/test_quickstart.py
"""
Tests for the quickstart command.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestQuickstartCommand:
    """Tests for fitz quickstart command."""

    def test_quickstart_shows_help(self):
        """Test that quickstart --help works."""
        result = runner.invoke(app, ["quickstart", "--help"])

        assert result.exit_code == 0
        assert "One-command RAG" in result.output
        assert "source" in result.output.lower()
        assert "question" in result.output.lower()


class TestResolveProvider:
    """Tests for provider auto-detection."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key-12345678"}, clear=False)
    def test_detects_cohere_from_env(self, tmp_path):
        """Test that Cohere is detected from environment variable."""
        from fitz_ai.cli.commands.quickstart import _resolve_provider

        config_path = tmp_path / "config.yaml"

        # Mock Ollama as not running
        with patch("fitz_ai.cli.commands.quickstart._check_ollama") as mock_ollama:
            mock_ollama.return_value = {"running": False, "ready": False}
            provider, reason, extra = _resolve_provider(config_path)

        assert provider == "cohere"
        assert "COHERE_API_KEY" in reason

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345678"}, clear=False)
    def test_detects_openai_from_env(self, tmp_path):
        """Test that OpenAI is detected from environment variable."""
        from fitz_ai.cli.commands.quickstart import _resolve_provider

        config_path = tmp_path / "config.yaml"

        # Mock Ollama as not running, and no Cohere key
        with (
            patch("fitz_ai.cli.commands.quickstart._check_ollama") as mock_ollama,
            patch.dict(os.environ, {"COHERE_API_KEY": ""}, clear=False),
        ):
            mock_ollama.return_value = {"running": False, "ready": False}
            # Clear COHERE_API_KEY for this test
            os.environ.pop("COHERE_API_KEY", None)
            provider, reason, extra = _resolve_provider(config_path)

        assert provider == "openai"
        assert "OPENAI_API_KEY" in reason

    def test_uses_existing_config(self, tmp_path):
        """Test that existing config is used when present."""
        from fitz_ai.cli.commands.quickstart import _resolve_provider

        config_path = tmp_path / "config.yaml"
        config_path.write_text("# existing config\n")

        provider, reason, extra = _resolve_provider(config_path)

        assert provider is None
        assert "existing configuration" in reason.lower()


class TestCheckOllama:
    """Tests for Ollama detection."""

    def test_returns_not_running_on_connection_error(self):
        """Test that Ollama is detected as not running on connection error."""
        from fitz_ai.cli.commands.quickstart import _check_ollama

        with patch("httpx.get") as mock_get:
            import httpx

            mock_get.side_effect = httpx.ConnectError("Connection refused")
            result = _check_ollama()

        assert result["running"] is False
        assert result["ready"] is False

    def test_detects_running_ollama_with_models(self):
        """Test that running Ollama with models is detected."""
        from fitz_ai.cli.commands.quickstart import _check_ollama

        mock_response = {
            "models": [
                {"name": "llama3.2:1b"},
                {"name": "nomic-embed-text:latest"},
            ]
        }

        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            result = _check_ollama()

        assert result["running"] is True
        assert result["ready"] is True
        assert result["chat_model"] == "llama3.2:1b"
        assert result["embedding_model"] == "nomic-embed-text:latest"

    def test_detects_missing_models(self):
        """Test that missing models are detected."""
        from fitz_ai.cli.commands.quickstart import _check_ollama

        mock_response = {"models": [{"name": "some-other-model:latest"}]}

        with patch("httpx.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_response
            result = _check_ollama()

        assert result["running"] is True
        assert result["ready"] is False
        assert len(result["missing"]) > 0


class TestCreateProviderConfig:
    """Tests for provider config generation."""

    def test_creates_config_file(self, tmp_path):
        """Test that config file is created."""
        from fitz_ai.cli.commands.quickstart import _create_provider_config

        config_path = tmp_path / "fitz.yaml"
        _create_provider_config(config_path, "cohere")

        assert config_path.exists()

    def test_config_has_required_sections(self, tmp_path):
        """Test that config has all required sections."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_provider_config

        config_path = tmp_path / "fitz.yaml"
        _create_provider_config(config_path, "cohere")

        config = yaml.safe_load(config_path.read_text())

        assert "chat" in config
        assert "embedding" in config
        assert "vector_db" in config
        assert "retrieval" in config
        assert "rerank" in config

    def test_config_uses_cohere(self, tmp_path):
        """Test that config uses Cohere when selected."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_provider_config

        config_path = tmp_path / "fitz.yaml"
        _create_provider_config(config_path, "cohere")

        config = yaml.safe_load(config_path.read_text())

        assert config["chat"]["plugin_name"] == "cohere"
        assert config["embedding"]["plugin_name"] == "cohere"
        assert config["rerank"]["plugin_name"] == "cohere"

    def test_config_uses_openai(self, tmp_path):
        """Test that config uses OpenAI when selected."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_provider_config

        config_path = tmp_path / "fitz.yaml"
        _create_provider_config(config_path, "openai")

        config = yaml.safe_load(config_path.read_text())

        assert config["chat"]["plugin_name"] == "openai"
        assert config["embedding"]["plugin_name"] == "openai"

    def test_config_uses_ollama(self, tmp_path):
        """Test that config uses Ollama when selected."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_provider_config

        config_path = tmp_path / "fitz.yaml"
        _create_provider_config(config_path, "ollama")

        config = yaml.safe_load(config_path.read_text())

        assert config["chat"]["plugin_name"] == "local_ollama"
        assert config["embedding"]["plugin_name"] == "local_ollama"

    def test_config_uses_local_faiss(self, tmp_path):
        """Test that config uses local FAISS."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_provider_config

        config_path = tmp_path / "fitz.yaml"
        _create_provider_config(config_path, "cohere")

        config = yaml.safe_load(config_path.read_text())

        assert config["vector_db"]["plugin_name"] == "local_faiss"

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        from fitz_ai.cli.commands.quickstart import _create_provider_config

        config_path = tmp_path / "nested" / "path" / "fitz.yaml"
        _create_provider_config(config_path, "cohere")

        assert config_path.exists()


class TestSaveApiKeyToEnv:
    """Tests for saving API key to environment config."""

    def test_saves_to_bashrc_on_unix(self, tmp_path):
        """Test saving to .bashrc on Unix."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_env

        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("# existing content\n")

        with (
            patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path),
            patch("platform.system", return_value="Linux"),
        ):
            _save_api_key_to_env("COHERE_API_KEY", "test-key-123")

        content = bashrc.read_text()
        assert "COHERE_API_KEY" in content
        assert "test-key-123" in content

    def test_saves_to_env_file_on_windows(self, tmp_path):
        """Test saving to .fitz/.env on Windows."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_env

        with (
            patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path),
            patch("platform.system", return_value="Windows"),
        ):
            _save_api_key_to_env("COHERE_API_KEY", "test-key-123")

        env_file = tmp_path / ".fitz" / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "COHERE_API_KEY" in content
        assert "test-key-123" in content

    def test_does_not_duplicate_on_unix(self, tmp_path):
        """Test that key is not duplicated if already present."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_env

        bashrc = tmp_path / ".bashrc"
        bashrc.write_text('export COHERE_API_KEY="existing-key"\n')

        with (
            patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path),
            patch("platform.system", return_value="Linux"),
        ):
            _save_api_key_to_env("COHERE_API_KEY", "new-key-123")

        content = bashrc.read_text()
        # Should still have old key, not new one
        assert "existing-key" in content
        assert content.count("COHERE_API_KEY") == 1
