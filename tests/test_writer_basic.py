from dataclasses import dataclass

from fitz_ai.vector_db.writer import VectorDBWriter, compute_chunk_hash


@dataclass
class DummyChunk:
    id: str
    doc_id: str
    chunk_index: int
    content: str
    metadata: dict


class DummyClient:
    def __init__(self):
        self.calls = []

    def upsert(self, collection, points, defer_persist: bool = False):
        self.calls.append((collection, points))


def test_compute_chunk_hash_is_deterministic():
    a = DummyChunk(id="1", doc_id="d", chunk_index=0, content="hello", metadata={"x": 1})
    b = DummyChunk(id="2", doc_id="d", chunk_index=0, content="hello", metadata={"x": 999})
    assert compute_chunk_hash(a) == compute_chunk_hash(b)


def test_compute_chunk_hash_changes_on_content_change():
    a = DummyChunk(id="1", doc_id="d", chunk_index=0, content="hello", metadata={})
    b = DummyChunk(id="1", doc_id="d", chunk_index=0, content="hello!", metadata={})
    assert compute_chunk_hash(a) != compute_chunk_hash(b)


def test_writer_upsert_builds_payload_and_calls_client():
    client = DummyClient()
    writer = VectorDBWriter(client=client)

    chunks = [
        DummyChunk(
            id="c1",
            doc_id="docA",
            chunk_index=0,
            content="alpha",
            metadata={"file": "a.txt"},
        ),
        DummyChunk(
            id="c2",
            doc_id="docA",
            chunk_index=1,
            content="beta",
            metadata={"file": "a.txt"},
        ),
    ]
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    writer.upsert(collection="col", chunks=chunks, vectors=vectors)

    assert len(client.calls) == 1
    collection, points = client.calls[0]
    assert collection == "col"
    assert len(points) == 2

    p0 = points[0]
    assert p0["id"] == "c1"
    assert p0["vector"] == [0.1, 0.2]
    assert p0["payload"]["doc_id"] == "docA"
    assert p0["payload"]["chunk_index"] == 0
    assert p0["payload"]["content"] == "alpha"
    assert p0["payload"]["metadata"]["file"] == "a.txt"
    assert "chunk_hash" in p0["payload"]
