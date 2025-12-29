# tests/test_ingestion_pipeline_end_to_end.py
from fitz_ai.core.chunk import Chunk
from fitz_ai.vector_db.writer import VectorDBWriter


class DummyVectorDB:
    def __init__(self):
        self.calls = []

    def upsert(self, collection, points, defer_persist: bool = False):
        self.calls.append((collection, points))


def test_ingestion_pipeline_end_to_end():
    vectordb = DummyVectorDB()
    writer = VectorDBWriter(client=vectordb)

    chunks = [
        Chunk(id="1", doc_id="doc1", chunk_index=0, content="A", metadata={"k": 1}),
        Chunk(id="2", doc_id="doc1", chunk_index=1, content="B", metadata={"k": 2}),
    ]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    writer.upsert(collection="col", chunks=chunks, vectors=vectors)

    assert len(vectordb.calls) == 1
    collection, points = vectordb.calls[0]
    assert collection == "col"
    assert len(points) == 2
    assert points[0]["id"] == "1"
    assert points[0]["payload"]["doc_id"] == "doc1"
    assert points[0]["payload"]["content"] == "A"
    assert "chunk_hash" in points[0]["payload"]
