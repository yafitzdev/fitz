# tests/test_smart_qdrant.py
"""
Tests for the Smart Qdrant Plugin.

These tests require a running Qdrant instance.
They will be skipped if Qdrant is not available.

To run with Qdrant:
    $env:QDRANT_HOST = "192.168.178.2"  # Your Qdrant host
    pytest tests/test_smart_qdrant.py -v

API Note:
    The Qdrant plugin (YAML-based) uses:
    - upsert(collection, points) where points is list of {id, vector, payload}
    - search(collection_name, query_vector, limit, ...) uses query_points internally
"""

import pytest

from fitz_ai.vector_db.loader import create_vector_db_plugin


def qdrant_available() -> bool:
    """Check if Qdrant is available."""
    try:
        db = create_vector_db_plugin("qdrant")
        db.list_collections()
        return True
    except Exception:
        return False


# Skip all tests in this module if Qdrant not available
pytestmark = pytest.mark.skipif(
    not qdrant_available(),
    reason="Qdrant not available (set QDRANT_HOST or start Qdrant on localhost:6333)",
)


@pytest.fixture
def qdrant_db():
    """Create a Qdrant client for testing."""
    return create_vector_db_plugin("qdrant")


@pytest.fixture
def test_collection_name():
    """Unique test collection name."""
    return "_fitz_pytest_auto_create"


@pytest.fixture(autouse=True)
def cleanup_collection(qdrant_db, test_collection_name):
    """Clean up test collection before and after each test."""
    # Cleanup before
    if test_collection_name in qdrant_db.list_collections():
        qdrant_db.delete_collection(test_collection_name)

    yield

    # Cleanup after
    if test_collection_name in qdrant_db.list_collections():
        qdrant_db.delete_collection(test_collection_name)


class TestQdrantConnection:
    """Test basic Qdrant connectivity."""

    def test_connection(self, qdrant_db):
        """Test that we can connect and list collections."""
        collections = qdrant_db.list_collections()
        assert isinstance(collections, list)

    def test_get_collection_stats(self, qdrant_db):
        """Test getting stats for an existing collection."""
        collections = qdrant_db.list_collections()
        if collections:
            stats = qdrant_db.get_collection_stats(collections[0])
            # Qdrant stats include status, points_count, etc.
            assert "status" in stats or "points_count" in stats


class TestAutoCreateCollection:
    """Test auto-creation of collections."""

    def test_auto_create_and_upsert(self, qdrant_db, test_collection_name):
        """Test that upserting to non-existent collection creates it."""
        # Upsert should auto-create - using the EXISTING API: upsert(collection, points)
        test_points = [
            {
                "id": "test-doc-1",
                "vector": [0.1] * 1024,
                "payload": {"content": "Test document", "doc_id": "doc1"},
            },
        ]

        qdrant_db.upsert(test_collection_name, test_points)

        # Verify collection was created
        assert test_collection_name in qdrant_db.list_collections()

        # Verify data was inserted
        stats = qdrant_db.get_collection_stats(test_collection_name)
        assert stats.get("points_count", 0) >= 1

    def test_search_after_upsert(self, qdrant_db, test_collection_name):
        """Test that search works after auto-create and upsert."""
        # Upsert - using the EXISTING API
        test_points = [
            {
                "id": "test-doc-1",
                "vector": [0.1] * 1024,
                "payload": {"content": "Test document about AI", "doc_id": "doc1"},
            },
            {
                "id": "test-doc-2",
                "vector": [0.2] * 1024,
                "payload": {"content": "Test document about ML", "doc_id": "doc2"},
            },
        ]

        qdrant_db.upsert(test_collection_name, test_points)

        # Search - using the EXISTING API
        results = qdrant_db.search(
            collection_name=test_collection_name,
            query_vector=[0.15] * 1024,
            limit=2,
        )

        assert len(results) == 2
        assert all(hasattr(r, "id") for r in results)
        assert all(hasattr(r, "score") for r in results)


class TestErrorMessages:
    """Test that error messages are helpful."""

    def test_collection_not_found_error(self, qdrant_db):
        """Test that searching non-existent collection gives helpful error."""
        import httpx

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            qdrant_db.search(
                collection_name="_nonexistent_collection_xyz",
                query_vector=[0.1] * 1024,
                limit=5,
            )

        # 404 status code for non-existent collection
        assert exc_info.value.response.status_code == 404


class TestStringIdConversion:
    """Test that string IDs are properly converted to UUIDs."""

    def test_string_id_converted(self, qdrant_db, test_collection_name):
        """Test that string IDs like 'doc.txt:0' work."""
        # Upsert with string ID (the problematic format)
        test_points = [
            {
                "id": "test_docs\\quantum.txt:0",  # This format caused issues before
                "vector": [0.1] * 1024,
                "payload": {"content": "Test", "doc_id": "doc1"},
            },
        ]

        # Should not raise
        qdrant_db.upsert(test_collection_name, test_points)

        # Verify it worked
        stats = qdrant_db.get_collection_stats(test_collection_name)
        assert stats.get("points_count", 0) >= 1

        # Original ID should be in payload
        results = qdrant_db.search(
            collection_name=test_collection_name,
            query_vector=[0.1] * 1024,
            limit=1,
        )
        assert len(results) >= 1
        assert results[0].payload.get("_original_id") == "test_docs\\quantum.txt:0"
