# tests/test_local_faiss_vector_db.py
"""
Tests for local FAISS vector database.

Tests the new lazy-initialization interface where dimension
is auto-detected on first upsert.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("faiss")

from fitz.backends.local_vector_db.faiss import FaissLocalVectorDB
from fitz.engines.classic_rag.models.chunk import Chunk


def _make_chunk(i: int, dim: int) -> Chunk:
    """Create a chunk with an embedding attached."""
    chunk = Chunk(
        id=f"c{i}",
        doc_id="doc",
        content=f"text {i}",
        chunk_index=i,
        metadata={"i": i},
    )

    object.__setattr__(
        chunk,
        "embedding",
        np.ones(dim, dtype="float32") * i,
    )

    return chunk


def test_local_faiss_vector_db_upsert_search_and_persist(tmp_path: Path):
    """Test upsert, search, and persistence with the new interface."""
    dim = 4

    # New interface: no dim required at init, just path
    db = FaissLocalVectorDB(path=tmp_path, persist=True)

    # Upsert points - dimension is auto-detected
    points = [
        {
            "id": f"c{i}",
            "vector": (np.ones(dim, dtype="float32") * i).tolist(),
            "payload": {"i": i, "content": f"text {i}"},
        }
        for i in range(5)
    ]
    db.upsert("test_collection", points)

    assert db.count() == 5
    assert db.count("test_collection") == 5

    # Search
    query = (np.ones(dim, dtype="float32") * 3).tolist()
    results = db.search(
        collection_name="test_collection",
        query_vector=query,
        limit=2,
    )

    assert len(results) == 2
    # Results should be sorted by similarity (closest to query vector * 3)
    assert results[0].id in {"c2", "c3", "c4"}
    assert "i" in results[0].payload

    # Persist and reload
    db._save()  # Explicit save (auto-save happens on upsert too)

    # Create new instance - should load from disk
    db_reloaded = FaissLocalVectorDB(path=tmp_path, persist=True)
    assert db_reloaded.count() == 5

    results_reloaded = db_reloaded.search(
        collection_name="test_collection",
        query_vector=query,
        limit=1,
    )

    assert results_reloaded[0].id == results[0].id


def test_local_faiss_vector_db_legacy_add_interface(tmp_path: Path):
    """Test the legacy add() interface with Chunk objects."""
    dim = 4

    db = FaissLocalVectorDB(path=tmp_path, persist=True)

    # Use legacy add() with chunks that have embeddings
    chunks = [_make_chunk(i, dim) for i in range(5)]
    db.add(chunks)

    assert db.count() == 5

    # Search using "default" collection (legacy add uses "default")
    query = (np.ones(dim, dtype="float32") * 3).tolist()
    results = db.search(
        collection_name="default",
        query_vector=query,
        limit=2,
    )

    assert len(results) == 2


def test_local_faiss_dimension_auto_detection(tmp_path: Path):
    """Test that dimension is correctly auto-detected."""
    db = FaissLocalVectorDB(path=tmp_path)

    # Initially no dimension
    assert db._dim is None

    # Upsert with 128-dim vectors
    points = [{"id": "1", "vector": [0.1] * 128, "payload": {}}]
    db.upsert("col", points)

    # Dimension should now be set
    assert db._dim == 128


def test_local_faiss_dimension_mismatch_error(tmp_path: Path):
    """Test that dimension mismatch raises clear error."""
    db = FaissLocalVectorDB(path=tmp_path)

    # First upsert sets dimension to 64
    points = [{"id": "1", "vector": [0.1] * 64, "payload": {}}]
    db.upsert("col", points)

    # Second upsert with different dimension should fail
    points_bad = [{"id": "2", "vector": [0.1] * 128, "payload": {}}]
    with pytest.raises(ValueError, match="Dimension mismatch"):
        db.upsert("col", points_bad)


def test_local_faiss_collection_filtering(tmp_path: Path):
    """Test that search filters by collection."""
    db = FaissLocalVectorDB(path=tmp_path)

    # Add to two collections
    db.upsert(
        "collection_a",
        [
            {"id": "a1", "vector": [1.0, 0.0, 0.0], "payload": {"name": "a1"}},
            {"id": "a2", "vector": [0.9, 0.1, 0.0], "payload": {"name": "a2"}},
        ],
    )

    db.upsert(
        "collection_b",
        [
            {"id": "b1", "vector": [0.0, 1.0, 0.0], "payload": {"name": "b1"}},
        ],
    )

    assert db.count() == 3
    assert db.count("collection_a") == 2
    assert db.count("collection_b") == 1

    # Search in collection_a
    results = db.search("collection_a", [1.0, 0.0, 0.0], limit=10)
    assert len(results) == 2
    assert all(r.id.startswith("a") for r in results)

    # Search in collection_b
    results = db.search("collection_b", [0.0, 1.0, 0.0], limit=10)
    assert len(results) == 1
    assert results[0].id == "b1"


def test_local_faiss_delete_collection(tmp_path: Path):
    """Test collection deletion."""
    db = FaissLocalVectorDB(path=tmp_path)

    db.upsert("keep", [{"id": "k1", "vector": [1.0, 0.0], "payload": {}}])
    db.upsert(
        "delete",
        [
            {"id": "d1", "vector": [0.0, 1.0], "payload": {}},
            {"id": "d2", "vector": [0.0, 0.9], "payload": {}},
        ],
    )

    assert db.count() == 3

    deleted = db.delete_collection("delete")
    assert deleted == 2
    assert db.count() == 1
    assert db.count("keep") == 1
    assert db.count("delete") == 0


def test_local_faiss_empty_search(tmp_path: Path):
    """Test search on empty database returns empty list."""
    db = FaissLocalVectorDB(path=tmp_path)

    results = db.search("any", [1.0, 0.0, 0.0], limit=10)
    assert results == []
