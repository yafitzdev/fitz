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
    EnrichmentContext,
    SummaryCache,
    ChunkSummarizer,
    EnrichmentPipeline,
    EnrichmentConfig,
)
from fitz_ai.ingest.enrichment.context.plugins.python import (
    Builder as PythonContextBuilder,
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


class TestSummaryCache:
    """Tests for SummaryCache."""

    def test_get_set_basic(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = SummaryCache(cache_path)

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
        cache1 = SummaryCache(cache_path)
        cache1.set("hash1", "enricher1", "description1")
        cache1.save()

        # Load from cache
        cache2 = SummaryCache(cache_path)
        assert cache2.get("hash1", "enricher1") == "description1"

    def test_context_manager(self, tmp_path):
        cache_path = tmp_path / "cache.json"

        with SummaryCache(cache_path) as cache:
            cache.set("hash1", "enricher1", "description1")

        # Should be saved on exit
        cache2 = SummaryCache(cache_path)
        assert cache2.get("hash1", "enricher1") == "description1"

    def test_clear(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = SummaryCache(cache_path)

        cache.set("hash1", "enricher1", "description1")
        cache.set("hash2", "enricher1", "description2")
        assert len(cache) == 2

        cache.clear()
        assert len(cache) == 0
        assert cache.get("hash1", "enricher1") is None

    def test_contains(self, tmp_path):
        cache_path = tmp_path / "cache.json"
        cache = SummaryCache(cache_path)

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


class TestChunkSummarizer:
    """Tests for ChunkSummarizer."""

    def test_basic_summarization(self, tmp_path):
        cache = SummaryCache(tmp_path / "cache.json")

        # Mock chat client
        mock_chat = MagicMock()
        mock_chat.chat.return_value = "This is a test description."

        summarizer = ChunkSummarizer(
            chat_client=mock_chat,
            cache=cache,
            enricher_id="test:v1",
        )

        description = summarizer.summarize(
            content="def hello(): pass",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )

        assert description == "This is a test description."
        mock_chat.chat.assert_called_once()

    def test_cache_hit(self, tmp_path):
        cache = SummaryCache(tmp_path / "cache.json")

        # Pre-populate cache
        cache.set("abc123", "test:v1", "Cached description")

        mock_chat = MagicMock()

        summarizer = ChunkSummarizer(
            chat_client=mock_chat,
            cache=cache,
            enricher_id="test:v1",
        )

        description = summarizer.summarize(
            content="def hello(): pass",
            file_path="/path/to/file.py",
            content_hash="abc123",
        )

        assert description == "Cached description"
        mock_chat.chat.assert_not_called()


class TestEnrichmentPipeline:
    """Tests for the EnrichmentPipeline."""

    def test_pipeline_creation(self, tmp_path):
        """Test that pipeline can be created with default config."""
        config = EnrichmentConfig(enabled=True)
        pipeline = EnrichmentPipeline(
            config=config,
            project_root=tmp_path,
            chat_client=None,
        )

        assert pipeline.is_enabled
        assert not pipeline.summaries_enabled  # No chat client
        assert pipeline.artifacts_enabled

    def test_pipeline_from_dict(self, tmp_path):
        """Test creating pipeline from dict config."""
        pipeline = EnrichmentPipeline.from_config(
            config={"enabled": True, "artifacts": {"auto": True}},
            project_root=tmp_path,
        )

        assert pipeline.is_enabled
        assert pipeline.artifacts_enabled

    def test_generate_structural_artifacts(self, tmp_path):
        """Test generating structural artifacts (no LLM required)."""
        # Create a simple Python project
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / "__init__.py").write_text("")
        (proj / "module.py").write_text('''
"""A simple module."""

class MyClass:
    """My class."""
    pass

def my_func():
    """My function."""
    pass
''')

        config = EnrichmentConfig(enabled=True)
        pipeline = EnrichmentPipeline(
            config=config,
            project_root=proj,
            chat_client=None,
        )

        artifacts = pipeline.generate_structural_artifacts()

        # Should generate structural artifacts (not architecture_narrative which needs LLM)
        assert len(artifacts) > 0
        artifact_names = [a.artifact_type.value for a in artifacts]
        assert "navigation_index" in artifact_names
        assert "interface_catalog" in artifact_names
        assert "architecture_narrative" not in artifact_names


class TestArtifactPluginDiscovery:
    """Tests for artifact plugin discovery."""

    def test_plugins_discovered(self):
        """Test that all expected plugins are discovered."""
        from fitz_ai.ingest.enrichment.artifacts.registry import get_artifact_registry

        registry = get_artifact_registry()
        plugin_names = registry.list_plugin_names()

        assert "navigation_index" in plugin_names
        assert "interface_catalog" in plugin_names
        assert "data_model_reference" in plugin_names
        assert "dependency_summary" in plugin_names
        assert "architecture_narrative" in plugin_names


class TestContextPluginDiscovery:
    """Tests for context plugin discovery."""

    def test_plugins_discovered(self):
        """Test that all expected plugins are discovered."""
        from fitz_ai.ingest.enrichment.context.registry import get_context_registry

        registry = get_context_registry()
        plugin_names = registry.list_plugin_names()

        assert "python" in plugin_names
        assert "generic" in plugin_names

    def test_python_extension_mapped(self):
        """Test that .py files are mapped to Python plugin."""
        from fitz_ai.ingest.enrichment.context.registry import get_context_registry

        registry = get_context_registry()
        plugin = registry.get_plugin_for_extension(".py")

        assert plugin is not None
        assert plugin.name == "python"
