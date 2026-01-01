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


class TestEnsureApiKey:
    """Tests for API key handling."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key-12345678"})
    def test_uses_existing_key(self):
        """Test that existing API key is used."""
        from fitz_ai.cli.commands.quickstart import _ensure_api_key

        key = _ensure_api_key("cohere")

        assert key == "test-key-12345678"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-12345678"})
    def test_uses_existing_openai_key(self):
        """Test that existing OpenAI API key is used."""
        from fitz_ai.cli.commands.quickstart import _ensure_api_key

        key = _ensure_api_key("openai")

        assert key == "sk-test-key-12345678"

    def test_ollama_returns_local(self):
        """Test that Ollama doesn't need an API key."""
        from fitz_ai.cli.commands.quickstart import _ensure_api_key

        key = _ensure_api_key("ollama")

        assert key == "local"


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


class TestSaveApiKeyToShell:
    """Tests for saving API key to shell config."""

    def test_saves_to_bashrc(self, tmp_path):
        """Test saving to .bashrc."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_shell

        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("# existing content\n")

        with patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path):
            _save_api_key_to_shell("test-key-123", "COHERE_API_KEY")

        content = bashrc.read_text()
        assert "COHERE_API_KEY" in content
        assert "test-key-123" in content

    def test_saves_openai_key(self, tmp_path):
        """Test saving OpenAI key to .bashrc."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_shell

        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("# existing content\n")

        with patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path):
            _save_api_key_to_shell("sk-test-key-123", "OPENAI_API_KEY")

        content = bashrc.read_text()
        assert "OPENAI_API_KEY" in content
        assert "sk-test-key-123" in content

    def test_does_not_duplicate(self, tmp_path):
        """Test that key is not duplicated if already present."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_shell

        bashrc = tmp_path / ".bashrc"
        bashrc.write_text('export COHERE_API_KEY="existing-key"\n')

        with patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path):
            _save_api_key_to_shell("new-key-123", "COHERE_API_KEY")

        content = bashrc.read_text()
        # Should still have old key, not new one
        assert "existing-key" in content
        assert content.count("COHERE_API_KEY") == 1
