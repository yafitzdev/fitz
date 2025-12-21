# tests/test_local_faiss_vector_db_default_config.py
"""
Tests for FAISS with default configuration.

Verifies that FaissLocalVectorDB works correctly with FitzPaths defaults.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("faiss")

from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB
from fitz_ai.core.paths import FitzPaths


def test_local_faiss_vector_db_uses_fitz_paths_default(tmp_path):
    """
    Verifies that:
    - FaissLocalVectorDB uses FitzPaths.vector_db() as default path
    - Persistence works without custom path overrides
    """
    # Override FitzPaths to use temp directory
    fake_workspace = tmp_path / ".fitz"

    with patch.object(FitzPaths, "workspace", return_value=fake_workspace):
        # Create DB with no path specified - should use FitzPaths default
        db = FaissLocalVectorDB()

        # Verify it's using the expected path
        expected_path = fake_workspace / "vector_db"
        assert db._base_path == expected_path

        # Add some data to trigger directory creation
        points = [
            {"id": "1", "vector": [1.0, 0.0, 0.0], "payload": {"text": "hello"}},
            {"id": "2", "vector": [0.0, 1.0, 0.0], "payload": {"text": "world"}},
            {"id": "3", "vector": [0.0, 0.0, 1.0], "payload": {"text": "test"}},
        ]
        db.upsert("default", points)

        # Directory should exist now
        assert expected_path.exists()

        # Verify persistence files were created
        assert (expected_path / "index.faiss").exists()
        assert (expected_path / "payloads.npy").exists()
        assert (expected_path / "dim.txt").exists()

        # Verify dimension was saved
        saved_dim = int((expected_path / "dim.txt").read_text())
        assert saved_dim == 3

        # Reload and verify data persisted
        db_reloaded = FaissLocalVectorDB()
        assert db_reloaded.count() == 3


def test_local_faiss_vector_db_custom_path_override(tmp_path):
    """Test that custom path overrides the default."""
    custom_path = tmp_path / "my_custom_vector_db"

    db = FaissLocalVectorDB(path=custom_path)

    assert db._base_path == custom_path

    # Add data
    db.upsert("test", [{"id": "1", "vector": [1.0, 2.0], "payload": {}}])

    # Should be in custom path
    assert custom_path.exists()
    assert (custom_path / "index.faiss").exists()


def test_local_faiss_persist_disabled(tmp_path):
    """Test that persist=False doesn't write to disk."""
    db = FaissLocalVectorDB(path=tmp_path, persist=False)

    db.upsert("test", [{"id": "1", "vector": [1.0, 2.0], "payload": {}}])

    # No files should be created (except maybe directory)
    assert not (tmp_path / "index.faiss").exists()

    # Data should still be in memory
    assert db.count() == 1


def test_local_faiss_reload_from_disk(tmp_path):
    """Test that data is correctly reloaded from disk."""
    # Create and populate
    db1 = FaissLocalVectorDB(path=tmp_path, persist=True)
    db1.upsert(
        "collection",
        [
            {
                "id": "doc1",
                "vector": [1.0, 0.0, 0.0, 0.0],
                "payload": {"title": "First"},
            },
            {
                "id": "doc2",
                "vector": [0.0, 1.0, 0.0, 0.0],
                "payload": {"title": "Second"},
            },
        ],
    )

    # Create new instance - should load from disk
    db2 = FaissLocalVectorDB(path=tmp_path, persist=True)

    assert db2.count() == 2
    assert db2._dim == 4

    # Search should work
    results = db2.search("collection", [1.0, 0.0, 0.0, 0.0], limit=1)
    assert len(results) == 1
    assert results[0].id == "doc1"
    assert results[0].payload["title"] == "First"
