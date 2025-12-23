# tests/test_chunking_router.py
"""
Tests for ChunkingRouter.

Verifies:
1. Router correctly routes documents to file-type specific chunkers
2. Fallback to default chunker works with warning
3. Configuration parsing and normalization works
4. Integration with ChunkingEngine works
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.plugins.simple import SimpleChunker
from fitz_ai.ingest.chunking.router import ChunkingRouter
from fitz_ai.ingest.chunking.engine import ChunkingEngine
from fitz_ai.engines.classic_rag.config import (
    ChunkingRouterConfig,
    ExtensionChunkerConfig,
)


# =============================================================================
# Mock Chunkers for Testing
# =============================================================================


@dataclass
class MockMarkdownChunker:
    """Mock markdown chunker for testing."""

    plugin_name: str = "markdown"
    max_tokens: int = 800

    @property
    def chunker_id(self) -> str:
        return f"{self.plugin_name}:{self.max_tokens}"

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        return [
            Chunk(
                id="md:0",
                doc_id=base_meta.get("doc_id", "test"),
                chunk_index=0,
                content=f"[MARKDOWN] {text[:50]}",
                metadata=base_meta,
            )
        ]


@dataclass
class MockPythonChunker:
    """Mock Python code chunker for testing."""

    plugin_name: str = "python_code"
    chunk_by: str = "function"

    @property
    def chunker_id(self) -> str:
        return f"{self.plugin_name}:{self.chunk_by}"

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        return [
            Chunk(
                id="py:0",
                doc_id=base_meta.get("doc_id", "test"),
                chunk_index=0,
                content=f"[PYTHON] {text[:50]}",
                metadata=base_meta,
            )
        ]


# =============================================================================
# Router Construction Tests
# =============================================================================


class TestRouterConstruction:
    """Tests for ChunkingRouter construction."""

    def test_basic_construction(self):
        """Router can be constructed with chunker map and default."""
        default = SimpleChunker(chunk_size=1000)
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=default,
        )

        assert router.default_chunker is default
        assert router.configured_extensions == []

    def test_construction_with_extensions(self):
        """Router tracks configured extensions."""
        default = SimpleChunker()
        md_chunker = MockMarkdownChunker()
        py_chunker = MockPythonChunker()

        router = ChunkingRouter(
            chunker_map={".md": md_chunker, ".py": py_chunker},
            default_chunker=default,
        )

        assert ".md" in router.configured_extensions
        assert ".py" in router.configured_extensions
        assert len(router.configured_extensions) == 2


# =============================================================================
# Routing Tests
# =============================================================================


class TestRouting:
    """Tests for document routing logic."""

    @pytest.fixture
    def router(self):
        """Create a router with md and py chunkers."""
        return ChunkingRouter(
            chunker_map={
                ".md": MockMarkdownChunker(),
                ".py": MockPythonChunker(),
            },
            default_chunker=SimpleChunker(chunk_size=1000),
            warn_on_fallback=True,
        )

    def test_routes_md_to_markdown_chunker(self, router):
        """Markdown files route to markdown chunker."""
        chunker = router.get_chunker(".md")
        assert chunker.plugin_name == "markdown"

    def test_routes_py_to_python_chunker(self, router):
        """Python files route to python chunker."""
        chunker = router.get_chunker(".py")
        assert chunker.plugin_name == "python_code"

    def test_unknown_extension_falls_back_to_default(self, router):
        """Unknown extensions fall back to default chunker."""
        chunker = router.get_chunker(".xyz")
        assert chunker.plugin_name == "simple"

    def test_txt_falls_back_to_default(self, router):
        """Unconfigured .txt falls back to default."""
        chunker = router.get_chunker(".txt")
        assert chunker.plugin_name == "simple"

    def test_extension_normalization(self, router):
        """Extensions are normalized to lowercase with dot."""
        assert router.get_chunker("md").plugin_name == "markdown"
        assert router.get_chunker(".MD").plugin_name == "markdown"
        assert router.get_chunker("Py").plugin_name == "python_code"

    def test_get_chunker_id(self, router):
        """get_chunker_id returns correct ID for extension."""
        assert router.get_chunker_id(".md") == "markdown:800"
        assert router.get_chunker_id(".py") == "python_code:function"
        assert router.get_chunker_id(".txt") == "simple:1000:0"


class TestFallbackWarning:
    """Tests for fallback warning behavior."""

    def test_warns_on_unknown_extension(self, caplog):
        """Logs warning when falling back to default."""
        router = ChunkingRouter(
            chunker_map={".md": MockMarkdownChunker()},
            default_chunker=SimpleChunker(),
            warn_on_fallback=True,
        )

        with caplog.at_level(logging.WARNING):
            router.get_chunker(".unknown")

        assert "No chunker configured for extension '.unknown'" in caplog.text
        assert "using default chunker" in caplog.text

    def test_warns_only_once_per_extension(self, caplog):
        """Only warns once per unknown extension."""
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=SimpleChunker(),
            warn_on_fallback=True,
        )

        with caplog.at_level(logging.WARNING):
            router.get_chunker(".xyz")
            router.get_chunker(".xyz")
            router.get_chunker(".xyz")

        assert caplog.text.count("No chunker configured") == 1

    def test_no_warning_when_disabled(self, caplog):
        """No warning when warn_on_fallback is False."""
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=SimpleChunker(),
            warn_on_fallback=False,
        )

        with caplog.at_level(logging.WARNING):
            router.get_chunker(".unknown")

        assert "No chunker configured" not in caplog.text


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRouterConfig:
    """Tests for ChunkingRouterConfig."""

    def test_extension_normalization_in_config(self):
        """Extensions in config are normalized."""
        config = ChunkingRouterConfig(
            default=ExtensionChunkerConfig(plugin_name="simple"),
            by_extension={
                "md": ExtensionChunkerConfig(plugin_name="markdown"),
                ".PY": ExtensionChunkerConfig(plugin_name="python"),
                ".Txt": ExtensionChunkerConfig(plugin_name="text"),
            },
        )

        assert ".md" in config.by_extension
        assert ".py" in config.by_extension
        assert ".txt" in config.by_extension


# =============================================================================
# Integration with ChunkingEngine
# =============================================================================


class TestEngineIntegration:
    """Tests for ChunkingEngine with router."""

    def test_engine_get_chunker_id(self):
        """Engine delegates get_chunker_id to router."""
        router = ChunkingRouter(
            chunker_map={".md": MockMarkdownChunker()},
            default_chunker=SimpleChunker(),
        )
        engine = ChunkingEngine(router=router)

        assert engine.get_chunker_id(".md") == "markdown:800"
        assert engine.get_chunker_id(".txt") == "simple:1000:0"


# =============================================================================
# Document Chunking Tests
# =============================================================================


class TestDocumentChunking:
    """Tests for chunking actual documents."""

    @dataclass
    class MockRawDocument:
        """Mock RawDocument for testing."""
        path: str
        content: str
        metadata: Dict[str, Any] = None

        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}

    def test_router_chunk_document(self):
        """chunk_document routes and chunks correctly."""
        router = ChunkingRouter(
            chunker_map={".md": MockMarkdownChunker()},
            default_chunker=SimpleChunker(chunk_size=100),
        )

        doc = self.MockRawDocument(
            path="/docs/readme.md",
            content="# Hello World\n\nThis is a test.",
        )

        chunks = router.chunk_document(doc)

        assert len(chunks) == 1
        assert "[MARKDOWN]" in chunks[0].content

    def test_router_chunk_document_uses_default(self):
        """chunk_document uses default for unknown extensions."""
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=SimpleChunker(chunk_size=100),
            warn_on_fallback=False,
        )

        doc = self.MockRawDocument(
            path="/docs/data.csv",
            content="a,b,c\n1,2,3",
        )

        chunks = router.chunk_document(doc)

        assert len(chunks) >= 1
        assert "[MARKDOWN]" not in chunks[0].content

    def test_engine_run_routes_by_extension(self):
        """Engine.run routes documents by extension."""
        router = ChunkingRouter(
            chunker_map={".md": MockMarkdownChunker()},
            default_chunker=SimpleChunker(chunk_size=100),
            warn_on_fallback=False,
        )
        engine = ChunkingEngine(router=router)

        md_doc = self.MockRawDocument(
            path="/docs/readme.md",
            content="# Markdown content",
        )
        txt_doc = self.MockRawDocument(
            path="/docs/notes.txt",
            content="Plain text content",
        )

        md_chunks = engine.run(md_doc)
        txt_chunks = engine.run(txt_doc)

        assert "[MARKDOWN]" in md_chunks[0].content
        assert "[MARKDOWN]" not in txt_chunks[0].content

    def test_empty_content_returns_empty_list(self):
        """Empty content produces no chunks."""
        router = ChunkingRouter(
            chunker_map={},
            default_chunker=SimpleChunker(),
            warn_on_fallback=False,
        )

        doc = self.MockRawDocument(path="/docs/empty.txt", content="   ")
        chunks = router.chunk_document(doc)

        assert chunks == []