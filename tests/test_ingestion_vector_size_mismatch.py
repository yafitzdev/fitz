import pytest
from fitz_ingest.ingester.engine import IngestionEngine
from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.chunker.plugins.simple import SimpleChunker
from fitz_ingest.exceptions.vector import IngestionVectorError

class BadEmbedder:
    def embed(self, text):
        return [0.1, 0.2]  # WRONG SIZE

class DummyClient:
    def upsert(self, *a, **k): pass
    def get_collections(self):
        class C: collections = []
        return C()
    def create_collection(self, *a, **k): pass

def test_ingestion_vector_size_mismatch(tmp_path):
    file = tmp_path / "f.txt"
    file.write_text("abc")

    engine = IngestionEngine(
        client=DummyClient(),
        collection="col",
        vector_size=4,
        chunker_engine=ChunkingEngine(SimpleChunker(chunk_size=5)),
        embedder=BadEmbedder(),
    )

    with pytest.raises(IngestionVectorError):
        engine.ingest_file(file)
