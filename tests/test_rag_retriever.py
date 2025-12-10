import pytest

from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.llm.embedding_client import DummyEmbeddingClient

from fitz_rag.config.schema import (
    RetrieverConfig,
    EmbeddingConfig,
    RerankConfig,
)


# ---------------------------------------------------------
# Minimal mock Qdrant client for retriever tests
# ---------------------------------------------------------
class MockQdrantSearchClient:
    """
    Emulates the minimal QdrantClient interface needed for retriever tests.
    Returns empty search results (no external DB needed).
    """
    def search(self, collection_name, query_vector, limit, with_payload=True):
        return []


def test_rag_retriever_import():
    from fitz_rag.retriever.dense_retriever import RAGRetriever
    assert RAGRetriever is not None


def test_rag_retriever_dummy(monkeypatch):
    # Use real provider name; later we'll patch embedder
    embed_cfg = EmbeddingConfig(
        provider="cohere",
        model="embed-english-v3.0",
        api_key=None,
    )

    retriever_cfg = RetrieverConfig(
        collection="test_collection",
        top_k=3,
        qdrant_host="localhost",
        qdrant_port=6333,
    )

    rerank_cfg = RerankConfig(
        provider="cohere",
        model="rerank-english-v3.0",
        api_key=None,
        enabled=False,
    )

    # Mock Qdrant client (no actual search backend)
    client = MockQdrantSearchClient()

    retriever = RAGRetriever(
        client=client,
        embed_cfg=embed_cfg,
        retriever_cfg=retriever_cfg,
        rerank_cfg=rerank_cfg,
    )

    # Patch dummy embedder after construction (avoids Cohere API call)
    retriever.embedder = DummyEmbeddingClient(dim=10)

    # Should run without error and return empty list
    result = retriever.retrieve("hello world")

    assert isinstance(result, list)
    assert len(result) == 0  # Mock client returns no results
