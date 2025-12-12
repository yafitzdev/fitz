from dataclasses import dataclass
from ingest.pipeline import IngestionPipeline


@dataclass
class DummyRaw:
    id: str
    text: str


@dataclass
class DummyChunk:
    id: str | None
    text: str
    metadata: dict


class DummyIngester:
    def run(self, source):
        return [DummyRaw(id="doc1", text="data")]


class DummyChunker:
    def run(self, raw_doc):
        return [
            DummyChunk(id=None, text="chunk1", metadata={"source": "doc1"}),
            DummyChunk(id=None, text="chunk1", metadata={"source": "doc1"}),  # duplicate text
        ]


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return [42.0]


class DummyVectorDB:
    def __init__(self):
        self.upserts = []

    def upsert(self, col, recs):
        self.upserts.append((col, recs))


def test_ingestion_pipeline_end_to_end():
    embedder = DummyEmbedder()
    vectordb = DummyVectorDB()

    # Writer performs dedupe + embed + write
    from core.vector_db.writer import VectorDBWriter
    writer = VectorDBWriter(embedder=embedder, vectordb=vectordb, deduplicate=True)

    pipeline = IngestionPipeline(
        ingester=DummyIngester(),
        chunker=DummyChunker(),
        writer=writer,
        collection="my_col",
    )

    written = pipeline.run("some_path")
    assert written == 1  # dedupe eliminates one chunk

    # Embedding called once
    assert embedder.calls == ["chunk1"]

    # Ensure vector DB received exactly one record
    assert len(vectordb.upserts) == 1
    col, recs = vectordb.upserts[0]
    assert col == "my_col"
    assert len(recs) == 1

    # Writer ensures chunk_id in payload
    assert "chunk_id" in recs[0].payload
    assert "chunk_hash" in recs[0].payload
