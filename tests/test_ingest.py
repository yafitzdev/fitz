"""
Tests for fitz_ingest ingestion flow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from fitz_ingest.chunker.plugins.simple import SimpleChunker
from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.ingester.engine import IngestionEngine
from fitz_ingest.vector_db.qdrant_utils import ensure_collection


# ---------------------------------------------------------
# Mock Qdrant client (no external DB needed)
# ---------------------------------------------------------
class MockQdrantClient:
    def __init__(self):
        self.collections = []
        self.upsert_calls = []

    # Simulates GET collections
    def get_collections(self):
        class C:
            collections = self.collections
        return C()

    # Simulates create_collection
    def create_collection(self, collection_name, vectors_config):
        self.collections.append(
            type("Obj", (), {"name": collection_name})
        )

    # Simulates upsert
    def upsert(self, collection_name, points):
        self.upsert_calls.append((collection_name, points))


# ---------------------------------------------------------
# Tests
# ---------------------------------------------------------
def test_simple_chunker(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("A" * 1200)

    plugin = SimpleChunker(chunk_size=500)
    engine = ChunkingEngine(plugin)

    chunks = engine.chunk_file(str(file_path))

    assert len(chunks) == 3
    assert chunks[0].text == "A" * 500
    assert chunks[1].text == "A" * 500
    assert chunks[2].text == "A" * 200


def test_ensure_collection():
    client = MockQdrantClient()

    ensure_collection(client, "test_collection", 1536)
    ensure_collection(client, "test_collection", 1536)  # should do nothing

    assert len(client.collections) == 1
    assert client.collections[0].name == "test_collection"


def test_ingestion_engine(tmp_path: Path):
    # Setup mock Qdrant
    client = MockQdrantClient()

    # Create a test file
    test_file = tmp_path / "doc.txt"
    test_file.write_text("hello world " * 100)  # ~1200 chars

    # Build plugin + chunking engine
    plugin = SimpleChunker(chunk_size=500)
    chunker_engine = ChunkingEngine(plugin)

    # Build ingestion engine
    engine = IngestionEngine(
        client=client,
        collection="my_col",
        vector_size=1536,
        chunker_engine=chunker_engine,
        embedder=None,
    )

    # Run ingestion
    engine.ingest_file(test_file)

    assert len(client.upsert_calls) == 1

    collection_name, points = client.upsert_calls[0]
    assert collection_name == "my_col"
    assert len(points) == 3  # 1200 chars => 500 + 500 + 200

    # Validate structure
    assert "text" in points[0]["payload"]
    assert "file" in points[0]["payload"]
