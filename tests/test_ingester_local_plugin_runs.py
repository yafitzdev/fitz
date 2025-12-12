from ingest.ingestion.engine import Ingester
from ingest.ingestion.base import RawDocument
from ingest.config.schema import IngestConfig


def test_ingester_local_plugin_runs(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("hello world")

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

    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].content == "hello world"
