# tests/unit/test_sdk_fitz.py
"""
Tests for the Fitz SDK.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest


def _write_test_config(path, collection="default"):
    """Write a minimal valid config file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"synthesizer: null\nquery_intelligence: null\ncollection: {collection}\n")


class TestFitzInit:
    """Tests for fitz initialization."""

    def test_default_collection(self):
        """Test default collection name."""
        from fitz_sage.sdk import fitz

        f = fitz()

        assert f.collection == "default"

    def test_custom_collection(self):
        """Test custom collection name."""
        from fitz_sage.sdk import fitz

        f = fitz(collection="my_collection")

        assert f.collection == "my_collection"

    def test_custom_config_path(self, tmp_path):
        """Test custom config path."""
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "custom.yaml"
        f = fitz(config_path=config_path)

        assert f.config_path == config_path

    def test_auto_init_default_true(self):
        """Test auto_init defaults to True."""
        from fitz_sage.sdk import fitz

        f = fitz()
        # auto_init is an internal attribute
        assert f._auto_init is True

    def test_concurrent_first_use_creates_one_engine(self, tmp_path, monkeypatch):
        """Concurrent SDK calls share one fully loaded engine."""
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path)
        engine = MagicMock()
        create_calls = 0
        creation_started = threading.Event()
        second_call_started = threading.Event()
        release_creation = threading.Event()

        def create_engine(**_kwargs):
            nonlocal create_calls
            create_calls += 1
            creation_started.set()
            assert release_creation.wait(timeout=2.0)
            return engine

        def get_engine(index):
            if index == 1:
                second_call_started.set()
            return client._get_engine()

        monkeypatch.setattr("fitz_sage.runtime.create_engine", create_engine)
        client = fitz(config_path=config_path)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(get_engine, 0)
            assert creation_started.wait(timeout=1.0)
            second = executor.submit(get_engine, 1)
            assert second_call_started.wait(timeout=1.0)
            release_creation.set()
            engines = [first.result(timeout=2.0), second.result(timeout=2.0)]

        assert engines == [engine, engine]
        assert create_calls == 1
        engine.load.assert_called_once_with("default")


class TestFitzConfigCreation:
    """Tests for config handling."""

    def test_ensure_config_skips_when_exists(self, tmp_path):
        """Test that _ensure_config does nothing when config exists."""
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path)
        f = fitz(config_path=config_path)

        f._ensure_config()  # Should not raise

        assert config_path.exists()

    def test_raises_without_auto_init(self, tmp_path):
        """Test that ConfigurationError is raised when auto_init=False and no config."""
        from fitz_sage.core import ConfigurationError
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "nonexistent.yaml"
        f = fitz(config_path=config_path, auto_init=False)

        with pytest.raises(ConfigurationError):
            f._ensure_config()


class TestFitzAnswer:
    """Tests for fitz.answer() method."""

    def test_raises_on_empty_question(self, tmp_path):
        """Test that QueryError is raised for empty question."""
        from fitz_sage.core import QueryError
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path)
        f = fitz(config_path=config_path)

        with pytest.raises(QueryError, match="cannot be empty"):
            f.answer("")

    def test_raises_on_whitespace_question(self, tmp_path):
        """Test that QueryError is raised for whitespace-only question."""
        from fitz_sage.core import QueryError
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "config.yaml"
        _write_test_config(config_path)
        f = fitz(config_path=config_path)

        with pytest.raises(QueryError, match="cannot be empty"):
            f.answer("   ")


class TestFitzEvidence:
    """Tests for fitz.evidence() method."""

    def test_evidence_points_source_and_returns_pack(self, tmp_path):
        """Evidence mode points an optional source before retrieval."""
        from fitz_sage.core import EvidencePack
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "config.yaml"
        source = tmp_path / "docs"
        source.mkdir()
        _write_test_config(config_path)

        expected = EvidencePack(query="question", mode=None)
        engine = MagicMock()
        engine.evidence.return_value = expected

        f = fitz(config_path=config_path)
        f._engine = engine

        result = f.evidence("question", source=source)

        assert result is expected
        engine.point.assert_called_once_with(source.resolve(), "default")
        engine.evidence.assert_called_once()


class TestFitzTrace:
    """Tests for fitz.trace() and governance replay."""

    def test_trace_points_source_and_returns_execution_record(self, tmp_path):
        from fitz_sage.core import RetrievalRun
        from fitz_sage.sdk import fitz

        config_path = tmp_path / "config.yaml"
        source = tmp_path / "docs"
        source.mkdir()
        _write_test_config(config_path)

        expected = MagicMock(spec=RetrievalRun)
        engine = MagicMock()
        engine.trace.return_value = expected
        f = fitz(config_path=config_path)
        f._engine = engine

        result = f.trace("question", source=source)

        assert result is expected
        engine.point.assert_called_once_with(source.resolve(), "default")
        engine.trace.assert_called_once()

    def test_trace_rejects_empty_question(self):
        from fitz_sage.core import QueryError
        from fitz_sage.sdk import fitz

        with pytest.raises(QueryError, match="cannot be empty"):
            fitz().trace(" ")


class TestFitzExports:
    """Tests for SDK exports."""

    def test_fitz_exported_from_sdk(self):
        """Test fitz is exported from fitz_sage.sdk."""
        from fitz_sage.sdk import fitz

        assert fitz is not None

    def test_fitz_exported_from_top_level(self):
        """Test fitz is exported from fitz_sage."""
        from fitz_sage import fitz

        assert fitz is not None

    def test_module_level_answer_exported_without_query_alias(self):
        """Synthesis uses the explicit answer() API without a query() alias."""
        import fitz_sage

        assert callable(fitz_sage.answer)
        assert not hasattr(fitz_sage, "query")

    def test_module_level_evidence_exported(self):
        """Test module-level evidence() is exported."""
        import fitz_sage

        assert hasattr(fitz_sage, "evidence")
        assert callable(fitz_sage.evidence)

    def test_trace_contracts_exported_from_top_level(self):
        """Execution records and module-level trace are public API."""
        import fitz_sage
        from fitz_sage import PyrrhoReplay, RetrievalRun

        assert RetrievalRun is not None
        assert PyrrhoReplay is not None
        assert callable(fitz_sage.trace)

    def test_evidence_types_exported_from_top_level(self):
        """Test evidence contracts are exported from fitz_sage."""
        from fitz_sage import EvidenceItem, EvidencePack

        assert EvidenceItem is not None
        assert EvidencePack is not None

    def test_module_level_evidence_delegates_to_default_fitz(self, monkeypatch):
        """Module-level evidence delegates to the default SDK instance."""
        import fitz_sage
        from fitz_sage.core import EvidencePack

        expected = EvidencePack(query="question", mode=None)
        sdk = MagicMock()
        sdk.evidence.return_value = expected
        monkeypatch.setattr(fitz_sage, "_get_default_fitz", lambda: sdk)

        result = fitz_sage.evidence("question", source="./docs")

        assert result is expected
        sdk.evidence.assert_called_once_with("question", source="./docs")

    def test_module_level_trace_delegates_to_default_fitz(self, monkeypatch):
        import fitz_sage
        from fitz_sage.core import RetrievalRun

        expected = MagicMock(spec=RetrievalRun)
        sdk = MagicMock()
        sdk.trace.return_value = expected
        monkeypatch.setattr(fitz_sage, "_get_default_fitz", lambda: sdk)

        result = fitz_sage.trace("question", source="./docs")

        assert result is expected
        sdk.trace.assert_called_once_with("question", source="./docs")
