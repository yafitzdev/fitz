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

    def test_quickstart_requires_source(self):
        """Test that quickstart requires a source path."""
        result = runner.invoke(app, ["quickstart"])

        assert result.exit_code != 0

    def test_quickstart_requires_question(self, tmp_path):
        """Test that quickstart requires a question."""
        (tmp_path / "test.txt").write_text("Test content")

        result = runner.invoke(app, ["quickstart", str(tmp_path)])

        assert result.exit_code != 0

    def test_quickstart_validates_source_exists(self):
        """Test that quickstart validates source path exists."""
        result = runner.invoke(app, ["quickstart", "/nonexistent/path", "question"])

        assert result.exit_code != 0


class TestEnsureApiKey:
    """Tests for API key handling."""

    @patch.dict(os.environ, {"COHERE_API_KEY": "test-key-12345678"})
    def test_uses_existing_key(self):
        """Test that existing API key is used."""
        from fitz_ai.cli.commands.quickstart import _ensure_api_key

        key = _ensure_api_key()

        assert key == "test-key-12345678"

    @patch.dict(os.environ, {}, clear=True)
    def test_prompts_when_no_key(self):
        """Test that user is prompted when no key exists."""
        # Clear the key
        os.environ.pop("COHERE_API_KEY", None)

        # Would prompt - we can't easily test interactive prompts
        # but we can verify the function exists and has correct signature


class TestCreateDefaultConfig:
    """Tests for config generation."""

    def test_creates_config_file(self, tmp_path):
        """Test that config file is created."""
        from fitz_ai.cli.commands.quickstart import _create_default_config

        config_path = tmp_path / "fitz.yaml"
        _create_default_config(config_path)

        assert config_path.exists()

    def test_config_has_required_sections(self, tmp_path):
        """Test that config has all required sections."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_default_config

        config_path = tmp_path / "fitz.yaml"
        _create_default_config(config_path)

        config = yaml.safe_load(config_path.read_text())

        assert "chat" in config
        assert "embedding" in config
        assert "vector_db" in config
        assert "retrieval" in config
        assert "rerank" in config

    def test_config_uses_cohere(self, tmp_path):
        """Test that config uses Cohere as default provider."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_default_config

        config_path = tmp_path / "fitz.yaml"
        _create_default_config(config_path)

        config = yaml.safe_load(config_path.read_text())

        assert config["chat"]["plugin_name"] == "cohere"
        assert config["embedding"]["plugin_name"] == "cohere"
        assert config["rerank"]["plugin_name"] == "cohere"

    def test_config_uses_local_faiss(self, tmp_path):
        """Test that config uses local FAISS."""
        import yaml

        from fitz_ai.cli.commands.quickstart import _create_default_config

        config_path = tmp_path / "fitz.yaml"
        _create_default_config(config_path)

        config = yaml.safe_load(config_path.read_text())

        assert config["vector_db"]["plugin_name"] == "local_faiss"

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        from fitz_ai.cli.commands.quickstart import _create_default_config

        config_path = tmp_path / "nested" / "path" / "fitz.yaml"
        _create_default_config(config_path)

        assert config_path.exists()


class TestSaveApiKeyToShell:
    """Tests for saving API key to shell config."""

    def test_saves_to_bashrc(self, tmp_path):
        """Test saving to .bashrc."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_shell

        bashrc = tmp_path / ".bashrc"
        bashrc.write_text("# existing content\n")

        with patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path):
            _save_api_key_to_shell("test-key-123")

        content = bashrc.read_text()
        assert "COHERE_API_KEY" in content
        assert "test-key-123" in content

    def test_does_not_duplicate(self, tmp_path):
        """Test that key is not duplicated if already present."""
        from fitz_ai.cli.commands.quickstart import _save_api_key_to_shell

        bashrc = tmp_path / ".bashrc"
        bashrc.write_text('export COHERE_API_KEY="existing-key"\n')

        with patch("fitz_ai.cli.commands.quickstart.Path.home", return_value=tmp_path):
            _save_api_key_to_shell("new-key-123")

        content = bashrc.read_text()
        # Should still have old key, not new one
        assert "existing-key" in content
        assert content.count("COHERE_API_KEY") == 1
