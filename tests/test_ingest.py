from fitz_rag.core import Chunk

class DummyChunker:
    def chunk_file(self, path: str):
        return [
            Chunk(id="1", text="hello world", metadata={"foo": "bar"}),
            Chunk(id="2", text="second chunk", metadata={}),
        ]


def test_ingestion_runs(monkeypatch):
    from fitz_ingest.ingester import IngestionEngine
    from fitz_rag.retriever.qdrant_client import create_qdrant_client

    client = create_qdrant_client()

    # Mock ensure_collection and upsert
    monkeypatch.setattr(
        client,
        "upsert",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        client,
        "get_collections",
        lambda: type("X", (), {"collections": []}),
    )
    monkeypatch.setattr(
        client,
        "create_collection",
        lambda **kwargs: None,
    )

    engine = IngestionEngine(
        client=client,
        collection="dummy",
        vector_size=10,
        embedder=None,
    )

    chunker = DummyChunker()

    engine.ingest_file(chunker, "dummy.txt")
