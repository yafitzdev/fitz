# tests/test_enrichment.py
"""Tests for the enrichment module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fitz_ai.ingest.enrichment import (
    CodeEnrichmentContext,
    ContentType,
    EnrichmentCache,
    EnrichmentContext,
    EnrichmentRouter,
    PythonContextBuilder,
    PythonProjectAnalyzer,
)


class TestEnrichmentContext:
    """Tests for EnrichmentContext and subclasses."""

    def test_base_context_defaults(self):
        ctx = EnrichmentContext(file_path="/path/to/file.py")
        assert ctx.file_path == "/path/to/file.py"
        assert ctx.content_type == ContentType.UNKNOWN
        assert ctx.file_extension == ".py"
        assert ctx.metadata == {}

    def test_code_context_python(self):
        ctx = CodeEnrichmentContext(
            file_path="/path/to/module.py",
            language="python",
            imports=["os", "sys"],
            exports=["class Foo", "def bar"],
            used_by=[("/path/to/other.py", "test")],
        )
        assert ctx.content_type == ContentType.CODE
        assert ctx.language == "python"
        assert len(ctx.imports) == 2
        assert len(ctx.exports) == 2
        assert len(ctx.used_by) == 1


class TestEnrichmentCache:
    """Tests for EnrichmentCache."""

    def test_get_set_basic(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = EnrichmentCache(cache_path)

        # Initially empty
        assert cache.get("hash1", "enricher1") is None

        # Set and get
        cache.set("hash1", "enricher1", "description1")
        assert cache.get("hash1", "enricher1") == "description1"

        # Different enricher_id should not match
        assert cache.get("hash1", "enricher2") is None

    def test_persistence(self, tmp_path):
        cache_path = tmp_path / "cache.json"

        # Write to cache
        cache1 = EnrichmentCache(cache_path)
        cache1.set("hash1", "enricher1", "description1")
        cache1.save()

        # Load from cache
        cache2 = EnrichmentCache(cache_path)
        assert cache2.get("hash1", "enricher1") == "description1"

    def test_context_manager(self, tmp_path):
        cache_path = tmp_path / "cache.json"

        with EnrichmentCache(cache_path) as cache:
            cache.set("hash1", "enricher1", "description1")

        # Should be saved on exit
        cache2 = EnrichmentCache(cache_path)
        assert cache2.get("hash1", "enricher1") == "description1"

    def test_clear(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = EnrichmentCache(cache_path)

        cache.set("hash1", "enricher1", "description1")
        cache.set("hash2", "enricher1", "description2")
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get("hash1", "enricher1") is None

    def test_contains(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = EnrichmentCache(cache_path)

        cache.set("hash1", "enricher1", "description1")
        assert ("hash1", "enricher1") in cache
        assert ("hash1", "enricher2") not in cache
        assert ("hash2", "enricher1") not in cache


class TestPythonContextBuilder:
    """Tests for PythonContextBuilder."""

    def test_quick_analyze_simple_file(self):
        builder = PythonContextBuilder()

        content = '''
"""Module docstring."""

import os
from pathlib import Path

class MyClass:
    pass

def my_function():
    pass
'''
        ctx = builder.build("/path/to/module.py", content)

        assert isinstance(ctx, CodeEnrichmentContext)
        assert ctx.content_type == ContentType.PYTHON
        assert ctx.language == "python"
        assert "os" in ctx.imports
        assert "pathlib" in ctx.imports
        assert "class MyClass" in ctx.exports
        assert "def my_function" in ctx.exports
        assert ctx.docstring == "Module docstring."

    def test_quick_analyze_syntax_error(self):
        builder = PythonContextBuilder()

        content = "def broken("  # Invalid syntax
        ctx = builder.build("/path/to/broken.py", content)

        # Should return basic context without crashing
        assert ctx.file_path == "/path/to/broken.py"

    def test_supported_extensions(self):
        builder = PythonContextBuilder()
        assert ".py" in builder.supported_extensions
        assert ".pyw" in builder.supported_extensions


class TestPythonProjectAnalyzer:
    """Tests for PythonProjectAnalyzer."""

    def test_analyze_simple_project(self, tmp_path):
        # Create a simple project structure
        pkg = tmp_path / "mypackage"
        pkg.mkdir()

        (pkg / "__init__.py").write_text("")
        (pkg / "module_a.py").write_text('''
"""Module A."""

def func_a():
    pass
''')
        (pkg / "module_b.py").write_text('''
"""Module B - imports module_a."""

from mypackage import module_a

def func_b():
    return module_a.func_a()
''')

        analyzer = PythonProjectAnalyzer(tmp_path)
        analyzer.analyze()

        # Check module_a analysis
        analysis_a = analyzer.get_analysis(str(pkg / "module_a.py"))
        assert analysis_a is not None
        assert "def func_a" in analysis_a.exports
        assert analysis_a.docstring == "Module A."

        # Check module_b analysis
        analysis_b = analyzer.get_analysis(str(pkg / "module_b.py"))
        assert analysis_b is not None
        assert "mypackage" in analysis_b.imports or "module_a" in str(analysis_b.imports)


class TestEnrichmentRouter:
    """Tests for EnrichmentRouter."""

    def test_basic_routing(self, tmp_path):
        cache = EnrichmentCache(tmp_path / "cache.json")

        # Mock chat client
        mock_chat = MagicMock()
        mock_chat.complete.return_value = "This is a test description."

        router = EnrichmentRouter(
            chat_client=mock_chat,
            cache=cache,
            enricher_id="test:v1",
        )

        description = router.enrich(
            content="def hello(): pass",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )

        assert description == "This is a test description."
        mock_chat.complete.assert_called_once()

    def test_cache_hit(self, tmp_path):
        cache = EnrichmentCache(tmp_path / "cache.json")

        # Pre-populate cache
        cache.set("abc123", "test:v1", "Cached description")

        mock_chat = MagicMock()

        router = EnrichmentRouter(
            chat_client=mock_chat,
            cache=cache,
            enricher_id="test:v1",
        )

        description = router.enrich(
            content="def hello(): pass",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )

        assert description == "Cached description"
        mock_chat.complete.assert_not_called()

    def test_python_context_integration(self, tmp_path):
        cache = EnrichmentCache(tmp_path / "cache.json")

        # Create a simple Python project
        proj = tmp_path / "project"
        proj.mkdir()
        (proj / "module.py").write_text("def hello(): pass")

        # Analyze project
        analyzer = PythonProjectAnalyzer(proj)
        analyzer.analyze()

        # Build router with Python support
        mock_chat = MagicMock()
        mock_chat.complete.return_value = "Python function description."

        router = EnrichmentRouter(
            chat_client=mock_chat,
            cache=cache,
            enricher_id="test:v1",
        )

        # Register Python builder
        builder = PythonContextBuilder(analyzer)
        router.register_builder(builder)

        description = router.enrich(
            content="def hello(): pass",
            file_path=str(proj / "module.py"),
            content_hash="xyz789",
        )

        assert description == "Python function description."

        # Check that the prompt includes Python-specific context
        call_args = mock_chat.complete.call_args
        prompt = call_args[0][0]
        assert "Python" in prompt
        assert "def hello" in prompt or "hello" in prompt


class TestEnrichmentIntegration:
    """Integration tests for the full enrichment pipeline."""

    def test_full_enrichment_flow(self, tmp_path):
        """Test the complete flow from project analysis to enrichment."""
        # Create project
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / "main.py").write_text('''
"""Main module."""

from myproject import utils

def main():
    return utils.helper()
''')
        (proj / "utils.py").write_text('''
"""Utility functions."""

def helper():
    return 42
''')

        # Analyze
        analyzer = PythonProjectAnalyzer(proj)
        analyzer.analyze()

        # Setup enrichment
        cache = EnrichmentCache(tmp_path / "cache.json")
        mock_chat = MagicMock()
        mock_chat.complete.side_effect = [
            "Main module that orchestrates the application.",
            "Utility module with helper functions.",
        ]

        router = EnrichmentRouter(
            chat_client=mock_chat,
            cache=cache,
            enricher_id="test:v1",
        )
        router.register_builder(PythonContextBuilder(analyzer))

        # Enrich both files
        desc1 = router.enrich(
            content=(proj / "main.py").read_text(),
            file_path=str(proj / "main.py"),
            content_hash="hash1",
        )
        desc2 = router.enrich(
            content=(proj / "utils.py").read_text(),
            file_path=str(proj / "utils.py"),
            content_hash="hash2",
        )

        assert "Main module" in desc1
        assert "Utility module" in desc2
        assert mock_chat.complete.call_count == 2

        # Save cache
        router.save_cache()

        # Verify cache persistence
        cache2 = EnrichmentCache(tmp_path / "cache.json")
        assert cache2.get("hash1", "test:v1") == desc1
        assert cache2.get("hash2", "test:v1") == desc2
