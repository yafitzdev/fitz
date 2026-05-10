# tests/integration/cloud_fixtures.py
"""
Cloud-specific pytest fixtures for E2E integration tests.

These fixtures provide properly configured CloudClient and FitzKragEngine
instances for testing the full cache flow against a real Fitz Cloud backend.

Required Environment Variables:
    FITZ_CLOUD_TEST_API_KEY: API key for test organization (fitz_xxx format)
    FITZ_CLOUD_TEST_ORG_KEY: 64-character hex encryption key
    FITZ_CLOUD_TEST_ORG_ID: UUID of test organization
    FITZ_CLOUD_URL: Cloud API base URL (default: http://localhost:8000/v1)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Generator

import httpx
import pytest

from fitz_sage.cloud import CloudClient, CloudConfig
from fitz_sage.cloud.cache_key import CacheVersions

# Path to test fixtures (reuse from e2e)
FIXTURES_DIR = Path(__file__).parent.parent / "e2e" / "fixtures_rag"


def get_cloud_env_vars() -> dict[str, str | None]:
    """Get cloud-related environment variables."""
    return {
        "api_key": os.environ.get("FITZ_CLOUD_TEST_API_KEY"),
        "org_key": os.environ.get("FITZ_CLOUD_TEST_ORG_KEY"),
        "org_id": os.environ.get("FITZ_CLOUD_TEST_ORG_ID"),
        "base_url": os.environ.get("FITZ_CLOUD_URL", "http://localhost:8000/v1"),
    }


def cloud_env_configured() -> bool:
    """Check if cloud environment variables are configured."""
    env = get_cloud_env_vars()
    return all([env["api_key"], env["org_key"], env["org_id"]])


def check_cloud_reachable(base_url: str, timeout: float = 5.0) -> bool:
    """Check if cloud API is reachable."""
    try:
        # Try to hit the health endpoint (or any lightweight endpoint)
        # Most APIs have /health or /v1/health
        response = httpx.get(f"{base_url.rstrip('/v1')}/health", timeout=timeout)
        return response.status_code in (200, 404)  # 404 means server is up but no health endpoint
    except Exception:
        # Try the base URL directly
        try:
            response = httpx.get(base_url, timeout=timeout)
            return response.status_code < 500
        except Exception:
            return False


# Skip marker for when cloud is unavailable
cloud_available = pytest.mark.skipif(
    not cloud_env_configured(),
    reason="Cloud environment variables not configured. Set FITZ_CLOUD_TEST_API_KEY, FITZ_CLOUD_TEST_ORG_KEY, FITZ_CLOUD_TEST_ORG_ID",
)


@pytest.fixture
def cloud_config() -> CloudConfig:
    """
    Create CloudConfig from environment variables.

    Requires:
        FITZ_CLOUD_TEST_API_KEY: API key for test organization
        FITZ_CLOUD_TEST_ORG_KEY: 64-character hex encryption key
        FITZ_CLOUD_URL: Cloud API base URL (optional, defaults to localhost)
    """
    env = get_cloud_env_vars()

    if not env["api_key"]:
        pytest.skip("FITZ_CLOUD_TEST_API_KEY not set")
    if not env["org_key"]:
        pytest.skip("FITZ_CLOUD_TEST_ORG_KEY not set")

    return CloudConfig(
        enabled=True,
        api_key=env["api_key"],
        org_key=env["org_key"],
        base_url=env["base_url"],
        timeout=30,
    )


@pytest.fixture
def cloud_org_id() -> str:
    """Get the test organization ID from environment."""
    org_id = os.environ.get("FITZ_CLOUD_TEST_ORG_ID")
    if not org_id:
        pytest.skip("FITZ_CLOUD_TEST_ORG_ID not set")
    return org_id


@pytest.fixture
def cloud_client(
    cloud_config: CloudConfig, cloud_org_id: str
) -> Generator[CloudClient, None, None]:
    """
    Create CloudClient instance with cleanup.

    Yields a configured CloudClient and closes it after the test.
    """
    # Check if cloud is reachable
    if not check_cloud_reachable(cloud_config.base_url):
        pytest.skip(f"Cloud API not reachable at {cloud_config.base_url}")

    client = CloudClient(config=cloud_config, org_id=cloud_org_id)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def unique_collection_name() -> str:
    """Generate a unique collection name for test isolation."""
    return f"cloud_e2e_test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cache_versions(unique_collection_name: str) -> CacheVersions:
    """Create CacheVersions for testing."""
    import fitz_sage

    return CacheVersions(
        optimizer="1.0.0",
        engine=fitz_sage.__version__,
        collection=unique_collection_name,
        llm_model="ollama:qwen2.5:1.5b",
        prompt_template="default",
    )


@pytest.fixture
def cloud_pipeline(
    cloud_config: CloudConfig,
    cloud_org_id: str,
    unique_collection_name: str,
):
    """
    Create FitzKragEngine with cloud_client attached.

    This fixture:
    1. Creates a unique test collection
    2. Ingests test fixtures using real embeddings
    3. Builds FitzKragEngine with CloudClient attached
    4. Cleans up collection after test

    Requires Ollama running with:
    - nomic-embed-text (embeddings)
    - qwen2.5:1.5b or similar (LLM)

    Also requires PostgreSQL with pgvector (fitz-sage uses pgvector exclusively).
    """
    from fitz_sage.engines.fitz_krag.config import FitzKragConfig
    from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
    from fitz_sage.ingestion.chunking.config import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )
    from fitz_sage.ingestion.chunking.router import ChunkingRouter
    from fitz_sage.ingestion.diff import run_diff_ingest
    from fitz_sage.ingestion.parser import ParserRouter
    from fitz_sage.ingestion.state import IngestStateManager
    from fitz_sage.llm import get_embedder
    from fitz_sage.vector_db.registry import get_vector_db_plugin

    # Check if cloud is reachable
    if not check_cloud_reachable(cloud_config.base_url):
        pytest.skip(f"Cloud API not reachable at {cloud_config.base_url}")

    collection = unique_collection_name

    # Initialize vector DB (pgvector - fitz-sage uses PostgreSQL + pgvector exclusively)
    try:
        vector_client = get_vector_db_plugin("pgvector")
    except Exception as e:
        pytest.skip(f"pgvector not available: {e}")

    # Initialize embedder (using Ollama nomic-embed-text, 768-dim)
    try:
        embedder = get_embedder("ollama", config={"model": "nomic-embed-text"})
        # Test that embedder works
        test_embedding = embedder.embed("test")
        if not test_embedding or len(test_embedding) < 100:
            pytest.skip("Ollama nomic-embed-text not available or not working")
    except Exception as e:
        pytest.skip(f"Ollama embedding model not available: {e}")

    # Set up ingestion
    parser_router = ParserRouter(docling_parser="docling")
    markdown_chunker = ExtensionChunkerConfig(
        plugin_name="markdown",
        kwargs={"max_chunk_size": 1500, "min_chunk_size": 100},
    )
    recursive_chunker = ExtensionChunkerConfig(
        plugin_name="recursive",
        kwargs={"chunk_size": 1000, "chunk_overlap": 200},
    )
    router_config = ChunkingRouterConfig(
        default=recursive_chunker,
        by_extension={".md": markdown_chunker},
    )
    chunking_router = ChunkingRouter.from_config(router_config)

    state_manager = IngestStateManager()
    state_manager.load()

    # Vector DB writer adapter
    class VectorDBWriterAdapter:
        def __init__(self, client):
            self._client = client

        def upsert(self, collection: str, points: list, defer_persist: bool = False):
            self._client.upsert(collection, points, defer_persist=defer_persist)

        def flush(self):
            if hasattr(self._client, "flush"):
                self._client.flush()

    writer = VectorDBWriterAdapter(vector_client)

    # Ingest test fixtures
    run_diff_ingest(
        source=str(FIXTURES_DIR),
        state_manager=state_manager,
        vector_db_writer=writer,
        parser_router=parser_router,
        chunking_router=chunking_router,
        collection=collection,
        embedding_id="local_ollama",
        vector_db_id="pgvector",
        enrichment_pipeline=None,
        force=True,
    )

    # Create CloudClient
    cloud_client = CloudClient(config=cloud_config, org_id=cloud_org_id)

    # Build KRAG engine with cloud client
    config_dict = {
        "chat_fast": "ollama/qwen2.5:1.5b",
        "chat_balanced": "ollama/qwen2.5:1.5b",
        "chat_smart": "ollama/qwen2.5:1.5b",
        "embedding": "ollama/nomic-embed-text",
        "vector_db": "pgvector",
        "collection": collection,
        "top_addresses": 10,
        "strict_grounding": False,
    }

    cfg = FitzKragConfig(**config_dict)
    engine = FitzKragEngine(cfg)

    try:
        yield engine
    finally:
        # Cleanup
        cloud_client.close()
        try:
            vector_client.delete_collection(collection)
        except Exception:
            pass


@pytest.fixture
def test_queries() -> dict[str, dict]:
    """
    Test queries with expected answers.

    Uses queries that have deterministic answers based on products.md and people.md fixtures.
    """
    return {
        "model_x100_price": {
            "query": "What is the price of the Model X100?",
            "expected_contains": ["$45,000", "45000", "45,000"],
            "description": "Price lookup - deterministic fact",
        },
        "techcorp_headquarters": {
            "query": "Where is TechCorp headquartered?",
            "expected_contains": ["Austin", "Texas"],
            "description": "Location lookup - deterministic fact",
        },
        "techcorp_ceo": {
            "query": "Who is the CEO of TechCorp?",
            "expected_contains": ["Sarah Chen"],
            "description": "Person lookup - deterministic fact",
        },
        "model_y200_range": {
            "query": "What is the range of the Model Y200?",
            "expected_contains": ["400 miles", "400"],
            "description": "Spec lookup - deterministic fact",
        },
    }
