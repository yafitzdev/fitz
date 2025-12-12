from dataclasses import dataclass

from ingest.pipeline import IngestionPipeline
from ingest.config.schema import IngestConfig


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
            DummyChunk(id=None, text="chunk1", metadata={"source": "doc1"}),
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


def test_ingestion_pipeline_end_to_end(monkeypatch):
    embedder = DummyEmbedder()
    vectordb = DummyVectorDB()

    from core.vector_db.writer import VectorDBWriter
    writer = VectorDBWriter(
        embedder=embedder,
        vectordb=vectordb,
        deduplicate=True,
    )

    # PATCH INGESTER REGISTRY (public)
    from ingest.ingestion import registry as ingest_registry
    monkeypatch.setitem(
        ingest_registry.REGISTRY,
        "dummy",
        lambda **_: DummyIngester(),
    )

    # PATCH CHUNKER FACTORY (public)
    from ingest.chunking import registry as chunker_registry
    monkeypatch.setattr(
        chunker_registry,
        "get_chunker_plugin",
        lambda name: (lambda **_: DummyChunker()),
    )

    cfg = IngestConfig(
        ingester={"plugin_name": "dummy"},
        chunker={"plugin_name": "dummy"},
        collection="my_col",
    )

    pipeline = IngestionPipeline(
        config=cfg,
        writer=writer,
    )

    written = pipeline.run("some_path")

    assert written == 1
    assert embedder.calls == ["chunk1"]

    col, recs = vectordb.upserts[0]
    assert col == "my_col"
    assert len(recs) == 1
    assert "chunk_id" in recs[0].payload
    assert "chunk_hash" in recs[0].payload
