# tests/test_cli_doctor.py
"""
Tests for the doctor command.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from fitz_ai.cli.cli import app

runner = CliRunner()


class TestDoctorCommand:
    """Tests for fitz doctor command."""

    def test_doctor_shows_help(self):
        """Test that doctor --help works."""
        result = runner.invoke(app, ["doctor", "--help"])

        assert result.exit_code == 0
        assert "doctor" in result.output.lower()
        assert "diagnostic" in result.output.lower()

    def test_doctor_runs_basic_checks(self):
        """Test that doctor runs without error."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.ollama.details = "not running"
        mock_system.qdrant.available = False
        mock_system.qdrant.details = "not running"
        mock_system.faiss.available = True
        mock_system.faiss.details = "installed"
        mock_system.api_keys = {}

        with (
            patch(
                "fitz_ai.cli.commands.doctor.detect_system_status",
                return_value=mock_system,
            ),
            patch(
                "fitz_ai.cli.context.FitzPaths.workspace",
                return_value=MagicMock(exists=lambda: False),
            ),
            patch(
                "fitz_ai.cli.context.FitzPaths.config",
                return_value=MagicMock(exists=lambda: False),
            ),
        ):
            result = runner.invoke(app, ["doctor"])

        # Should complete even without config
        assert "python" in result.output.lower()


class TestDoctorChecks:
    """Tests for individual doctor check functions."""

    def test_check_python_version(self):
        """Test _check_python returns correct info."""
        from fitz_ai.cli.commands.doctor import _check_python

        ok, detail = _check_python()

        assert isinstance(ok, bool)
        assert "Python" in detail
        assert str(sys.version_info.major) in detail

    def test_check_python_version_ok(self):
        """Test _check_python returns True for valid version."""
        from fitz_ai.cli.commands.doctor import _check_python

        ok, _ = _check_python()

        # We're running Python 3.10+, so should be ok
        if sys.version_info >= (3, 10):
            assert ok is True

    def test_check_workspace_exists(self, tmp_path):
        """Test _check_workspace when workspace exists."""
        workspace = tmp_path / ".fitz"
        workspace.mkdir()

        with patch(
            "fitz_ai.cli.context.FitzPaths.workspace",
            return_value=workspace,
        ):
            from fitz_ai.cli.commands.doctor import _check_workspace

            ok, detail = _check_workspace()

        assert ok is True
        assert str(workspace) in detail

    def test_check_workspace_missing(self, tmp_path):
        """Test _check_workspace when workspace is missing."""
        workspace = tmp_path / ".fitz"

        with patch(
            "fitz_ai.cli.context.FitzPaths.workspace",
            return_value=workspace,
        ):
            from fitz_ai.cli.commands.doctor import _check_workspace

            ok, detail = _check_workspace()

        assert ok is False
        assert "init" in detail.lower()

    def test_check_config_valid(self, tmp_path):
        """Test _check_config with valid config."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {"chat": {"plugin_name": "cohere"}}
        config_path.write_text(yaml.dump(config))

        with (
            patch("fitz_ai.cli.context.FitzPaths.engine_config", return_value=config_path),
            patch("fitz_ai.cli.context.FitzPaths.config", return_value=config_path),
        ):
            from fitz_ai.cli.commands.doctor import _check_config

            ok, detail, ctx = _check_config()

        assert ok is True
        assert detail == "Valid"
        assert ctx.chat_plugin == "cohere"

    def test_check_config_missing(self, tmp_path):
        """Test _check_config with missing config."""
        config_path = tmp_path / "missing.yaml"
        engine_path = tmp_path / "fitz_rag.yaml"

        with (
            patch("fitz_ai.cli.context.FitzPaths.engine_config", return_value=engine_path),
            patch("fitz_ai.cli.context.FitzPaths.config", return_value=config_path),
        ):
            from fitz_ai.cli.commands.doctor import _check_config

            ok, detail, ctx = _check_config()

        assert ok is False
        assert "init" in detail.lower()
        assert ctx is None

    def test_check_config_invalid(self, tmp_path):
        """Test _check_config with invalid YAML."""
        config_path = tmp_path / "fitz.yaml"
        config_path.write_text("invalid: yaml: content: :")

        with (
            patch("fitz_ai.cli.context.FitzPaths.engine_config", return_value=config_path),
            patch("fitz_ai.cli.context.FitzPaths.config", return_value=config_path),
        ):
            from fitz_ai.cli.commands.doctor import _check_config

            ok, detail, ctx = _check_config()

        # CLIContext.load_or_none returns None for invalid config
        assert ok is False
        assert ctx is None


class TestDoctorDependencies:
    """Tests for dependency checks."""

    def test_check_dependencies_finds_typer(self):
        """Test _check_dependencies finds typer."""
        from fitz_ai.cli.commands.doctor import _check_dependencies

        results = _check_dependencies()

        typer_result = next((r for r in results if r[0] == "typer"), None)
        assert typer_result is not None
        assert typer_result[1] is True  # installed

    def test_check_dependencies_finds_pydantic(self):
        """Test _check_dependencies finds pydantic."""
        from fitz_ai.cli.commands.doctor import _check_dependencies

        results = _check_dependencies()

        pydantic_result = next((r for r in results if r[0] == "pydantic"), None)
        assert pydantic_result is not None
        assert pydantic_result[1] is True

    def test_check_optional_dependencies(self):
        """Test _check_optional_dependencies returns results."""
        from fitz_ai.cli.commands.doctor import _check_optional_dependencies

        results = _check_optional_dependencies()

        assert len(results) > 0
        assert all(len(r) == 3 for r in results)  # (name, ok, detail)


class TestDoctorConnectionTests:
    """Tests for connection test functions."""

    def _make_ctx(self, **kwargs):
        """Create a CLIContext for testing."""
        from fitz_ai.cli.context import CLIContext

        return CLIContext(
            raw_config={},
            chat_plugin=kwargs.get("chat_plugin", ""),
            embedding_plugin=kwargs.get("embedding_plugin", ""),
            vector_db_plugin=kwargs.get("vector_db_plugin", ""),
            vector_db_kwargs=kwargs.get("vector_db_kwargs", {}),
            rerank_enabled=kwargs.get("rerank_enabled", False),
            rerank_plugin=kwargs.get("rerank_plugin", ""),
        )

    def test_test_embedding_not_configured(self):
        """Test _test_embedding when not configured."""
        from fitz_ai.cli.commands.doctor import _test_embedding

        ctx = self._make_ctx(embedding_plugin="")
        ok, detail = _test_embedding(ctx)

        assert ok is False
        assert "not configured" in detail.lower()

    def test_test_embedding_success(self):
        """Test _test_embedding with working plugin."""
        mock_plugin = MagicMock()
        mock_plugin.embed.return_value = [0.1, 0.2, 0.3]

        with patch(
            "fitz_ai.llm.registry.get_llm_plugin",
            return_value=mock_plugin,
        ):
            from fitz_ai.cli.commands.doctor import _test_embedding

            ctx = self._make_ctx(embedding_plugin="cohere")
            ok, detail = _test_embedding(ctx)

        assert ok is True
        assert "dim=3" in detail

    def test_test_chat_not_configured(self):
        """Test _test_chat when not configured."""
        from fitz_ai.cli.commands.doctor import _test_chat

        ctx = self._make_ctx(chat_plugin="")
        ok, detail = _test_chat(ctx)

        assert ok is False
        assert "not configured" in detail.lower()

    def test_test_chat_success(self):
        """Test _test_chat with working plugin."""
        mock_plugin = MagicMock()

        with patch(
            "fitz_ai.llm.registry.get_llm_plugin",
            return_value=mock_plugin,
        ):
            from fitz_ai.cli.commands.doctor import _test_chat

            ctx = self._make_ctx(chat_plugin="cohere")
            ok, detail = _test_chat(ctx)

        assert ok is True
        assert "ready" in detail.lower()

    def test_test_vector_db_not_configured(self):
        """Test _test_vector_db when not configured."""
        from fitz_ai.cli.commands.doctor import _test_vector_db

        ctx = self._make_ctx(vector_db_plugin="")
        ok, detail = _test_vector_db(ctx)

        assert ok is False
        assert "not configured" in detail.lower()

    def test_test_vector_db_success(self):
        """Test _test_vector_db with working plugin."""
        mock_plugin = MagicMock()
        mock_plugin.list_collections.return_value = ["coll1", "coll2"]

        with patch(
            "fitz_ai.vector_db.registry.get_vector_db_plugin",
            return_value=mock_plugin,
        ):
            from fitz_ai.cli.commands.doctor import _test_vector_db

            ctx = self._make_ctx(vector_db_plugin="local_faiss")
            ok, detail = _test_vector_db(ctx)

        assert ok is True
        assert "2 collections" in detail

    def test_test_rerank_disabled(self):
        """Test _test_rerank when disabled."""
        from fitz_ai.cli.commands.doctor import _test_rerank

        ctx = self._make_ctx(rerank_enabled=False)
        ok, detail = _test_rerank(ctx)

        assert ok is True
        assert "disabled" in detail.lower()

    def test_test_rerank_enabled_success(self):
        """Test _test_rerank when enabled and working."""
        mock_plugin = MagicMock()

        with patch(
            "fitz_ai.llm.registry.get_llm_plugin",
            return_value=mock_plugin,
        ):
            from fitz_ai.cli.commands.doctor import _test_rerank

            ctx = self._make_ctx(rerank_enabled=True, rerank_plugin="cohere")
            ok, detail = _test_rerank(ctx)

        assert ok is True
        assert "ready" in detail.lower()


class TestDoctorVerboseMode:
    """Tests for verbose mode."""

    def test_doctor_verbose_shows_more(self):
        """Test --verbose shows additional information."""
        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.ollama.details = "not running"
        mock_system.qdrant.available = False
        mock_system.qdrant.details = "not running"
        mock_system.faiss.available = True
        mock_system.faiss.details = "installed"
        mock_system.api_keys = {}

        with (
            patch(
                "fitz_ai.cli.commands.doctor.detect_system_status",
                return_value=mock_system,
            ),
            patch(
                "fitz_ai.cli.context.FitzPaths.workspace",
                return_value=MagicMock(exists=lambda: True),
            ),
            patch(
                "fitz_ai.cli.context.FitzPaths.config",
                return_value=MagicMock(exists=lambda: False),
            ),
        ):
            result = runner.invoke(app, ["doctor", "-v"])

        # Verbose should show optional packages section
        assert "optional" in result.output.lower()


class TestDoctorTestMode:
    """Tests for test mode."""

    def test_doctor_test_runs_connections(self, tmp_path):
        """Test --test runs connection tests."""
        import yaml

        config_path = tmp_path / "fitz.yaml"
        config = {
            "chat": {"plugin_name": "cohere"},
            "embedding": {"plugin_name": "cohere"},
            "vector_db": {"plugin_name": "local_faiss"},
            "rerank": {"enabled": False},
        }
        config_path.write_text(yaml.dump(config))

        mock_system = MagicMock()
        mock_system.ollama.available = False
        mock_system.ollama.details = "not running"
        mock_system.qdrant.available = False
        mock_system.qdrant.details = "not running"
        mock_system.faiss.available = True
        mock_system.faiss.details = "installed"
        mock_system.api_keys = {}

        mock_plugin = MagicMock()
        mock_plugin.embed.return_value = [0.1, 0.2]
        mock_plugin.list_collections.return_value = []

        with (
            patch(
                "fitz_ai.cli.commands.doctor.detect_system_status",
                return_value=mock_system,
            ),
            patch("fitz_ai.cli.context.FitzPaths.workspace", return_value=tmp_path),
            patch("fitz_ai.cli.context.FitzPaths.config", return_value=config_path),
            patch("fitz_ai.llm.registry.get_llm_plugin", return_value=mock_plugin),
            patch(
                "fitz_ai.vector_db.registry.get_vector_db_plugin",
                return_value=mock_plugin,
            ),
        ):
            result = runner.invoke(app, ["doctor", "--test"])

        # Should show connection test section
        assert "connection" in result.output.lower() or "embedding" in result.output.lower()
