from ingest.ingestion.engine import Ingester
from ingest.ingestion.base import RawDocument
from ingest.config.schema import IngestConfig


def test_ingester_returns_raw_documents(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("sample")

    cfg = IngestConfig(
        ingester={
            "plugin_name": "local",
        },
        chunker={
            "plugin_name": "simple",
            "chunk_size": 1000,
            "chunk_overlap": 0,
        },
        collection="test",
    )

    ingester = Ingester(config=cfg)
    docs = list(ingester.run(str(tmp_path)))

    doc = docs[0]
    assert isinstance(doc, RawDocument)
    assert doc.path.endswith("sample.txt")
    assert doc.metadata["source"] == "local_fs"
