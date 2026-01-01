# tests/engines/fitz_rag/retrieval/test_yaml_plugins.py
"""
Tests for YAML-based retrieval plugins.

These tests verify that:
1. YAML plugin files are discovered
2. Plugin specs are loaded correctly
3. Pipelines are built from specs
4. Steps execute correctly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from fitz_ai.core.chunk import Chunk
from fitz_ai.engines.fitz_rag.retrieval.loader import (
    RetrievalDependencies,
    StepSpec,
    build_pipeline_from_spec,
    create_retrieval_pipeline,
    list_available_plugins,
    load_plugin_spec,
)
from fitz_ai.engines.fitz_rag.retrieval.registry import (
    available_retrieval_plugins,
    get_retrieval_plugin,
)

# =============================================================================
# Mock Dependencies
# =============================================================================


@dataclass
class MockHit:
    """Mock vector search hit."""

    id: str
    score: float
    payload: dict[str, Any]


class MockVectorClient:
    """Mock vector database client."""

    def __init__(self, hits: list[MockHit] | None = None):
        self.hits = hits or []
        self.search_calls: list[dict] = []

    def search(
        self,
        collection_name: str = "",
        query_vector: list[float] | None = None,
        limit: int = 10,
        with_payload: bool = True,
        query_filter: dict | None = None,
    ) -> list[MockHit]:
        self.search_calls.append(
            {
                "collection": collection_name,
                "vector": query_vector,
                "limit": limit,
                "filter": query_filter,
            }
        )
        return self.hits[:limit]


class MockEmbedder:
    """Mock embedding service."""

    def __init__(self, vector: list[float] | None = None):
        self.vector = vector or [0.1, 0.2, 0.3]
        self.embed_calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        return self.vector


class MockReranker:
    """Mock reranking service."""

    def __init__(self):
        self.rerank_calls: list[dict] = []

    def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        self.rerank_calls.append({"query": query, "documents": documents, "top_n": top_n})
        # Return in original order with descending scores
        # Start at 0.99 to avoid triggering VIP handling (score=1.0)
        n = top_n or len(documents)
        results = [(i, 0.99 - i * 0.1) for i in range(min(n, len(documents)))]
        return results


# =============================================================================
# Test Data
# =============================================================================


def make_hits(n: int) -> list[MockHit]:
    """Create test vector search hits."""
    return [
        MockHit(
            id=f"hit_{i}",
            score=0.95 - i * 0.05,
            payload={
                "doc_id": f"doc_{i % 3}",
                "content": f"Content of hit {i}",
                "chunk_index": i,
            },
        )
        for i in range(n)
    ]


# =============================================================================
# Tests: Plugin Discovery
# =============================================================================


class TestPluginDiscovery:
    def test_list_available_plugins(self):
        """Should discover YAML plugin files."""
        plugins = list_available_plugins()
        assert "dense" in plugins

    def test_registry_list_matches_loader(self):
        """Registry and loader should return same plugins."""
        loader_plugins = list_available_plugins()
        registry_plugins = available_retrieval_plugins()
        assert set(loader_plugins) == set(registry_plugins)


# =============================================================================
# Tests: Plugin Spec Loading
# =============================================================================


class TestPluginSpecLoading:
    def test_load_dense_spec(self):
        """Should load dense.yaml plugin spec."""
        spec = load_plugin_spec("dense")

        assert spec.plugin_name == "dense"
        assert len(spec.steps) > 0
        # First step in raw spec is artifact_fetch (conditionally enabled)
        assert spec.steps[0].type == "artifact_fetch"
        assert spec.steps[0].enabled_if == "fetch_artifacts"
        # Vector search is second
        assert spec.steps[1].type == "vector_search"

    def test_step_spec_from_dict(self):
        """Should parse step spec from dict."""
        data = {"type": "vector_search", "k": 25}
        step = StepSpec.from_dict(data)

        assert step.type == "vector_search"
        assert step.params["k"] == 25

    def test_step_spec_with_enabled_if(self):
        """Should parse enabled_if condition."""
        data = {"type": "rerank", "k": 10, "enabled_if": "reranker"}
        step = StepSpec.from_dict(data)

        assert step.type == "rerank"
        assert step.enabled_if == "reranker"

    def test_load_nonexistent_plugin_raises(self):
        """Should raise error for missing plugin."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_plugin_spec("nonexistent_plugin")


# =============================================================================
# Tests: Pipeline Building
# =============================================================================


class TestPipelineBuilding:
    def test_build_pipeline_from_dense_spec(self):
        """Should build pipeline from dense spec."""
        spec = load_plugin_spec("dense")
        deps = RetrievalDependencies(
            vector_client=MockVectorClient(make_hits(10)),
            embedder=MockEmbedder(),
            collection="test_collection",
            reranker=None,
            top_k=5,
        )

        steps = build_pipeline_from_spec(spec, deps)

        # Should have at least vector_search and limit
        assert len(steps) >= 2
        assert steps[0].name == "VectorSearchStep"

    def test_build_pipeline_skips_rerank_without_reranker(self):
        """Should skip rerank step if reranker not provided."""
        spec = load_plugin_spec("dense")
        deps = RetrievalDependencies(
            vector_client=MockVectorClient(make_hits(10)),
            embedder=MockEmbedder(),
            collection="test_collection",
            reranker=None,  # No reranker
            top_k=5,
        )

        steps = build_pipeline_from_spec(spec, deps)

        # Should NOT include rerank step
        step_names = [s.name for s in steps]
        assert "RerankStep" not in step_names

    def test_build_pipeline_includes_rerank_with_reranker(self):
        """Should include rerank step if reranker provided."""
        spec = load_plugin_spec("dense")
        deps = RetrievalDependencies(
            vector_client=MockVectorClient(make_hits(10)),
            embedder=MockEmbedder(),
            collection="test_collection",
            reranker=MockReranker(),  # With reranker
            top_k=5,
        )

        steps = build_pipeline_from_spec(spec, deps)

        # Should include rerank step
        step_names = [s.name for s in steps]
        assert "RerankStep" in step_names


# =============================================================================
# Tests: Pipeline Execution
# =============================================================================


class TestPipelineExecution:
    def test_create_and_retrieve(self):
        """Should create pipeline and execute retrieval."""
        pipeline = create_retrieval_pipeline(
            plugin_name="dense",
            vector_client=MockVectorClient(make_hits(10)),
            embedder=MockEmbedder(),
            collection="test_collection",
            reranker=None,
            top_k=5,
        )

        chunks = pipeline.retrieve("test query")

        assert len(chunks) == 5
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_retrieve_with_reranker(self):
        """Should execute retrieval with reranking."""
        reranker = MockReranker()
        pipeline = create_retrieval_pipeline(
            plugin_name="dense",
            vector_client=MockVectorClient(make_hits(20)),
            embedder=MockEmbedder(),
            collection="test_collection",
            reranker=reranker,
            top_k=5,
        )

        chunks = pipeline.retrieve("test query")

        # Reranker should have been called
        assert len(reranker.rerank_calls) == 1
        assert len(chunks) == 5


# =============================================================================
# Tests: Registry
# =============================================================================


class TestRegistry:
    def test_get_retrieval_plugin(self):
        """Should get plugin via registry."""
        pipeline = get_retrieval_plugin(
            plugin_name="dense",
            vector_client=MockVectorClient(make_hits(10)),
            embedder=MockEmbedder(),
            collection="test_collection",
        )

        assert pipeline.plugin_name == "dense"

    def test_get_nonexistent_plugin_raises(self):
        """Should raise PluginNotFoundError for missing plugin."""
        from fitz_ai.engines.fitz_rag.retrieval.registry import (
            PluginNotFoundError,
        )

        with pytest.raises(PluginNotFoundError):
            get_retrieval_plugin(
                plugin_name="nonexistent",
                vector_client=MockVectorClient(),
                embedder=MockEmbedder(),
                collection="test",
            )
