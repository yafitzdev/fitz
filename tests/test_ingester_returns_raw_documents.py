from fitz_ai.ingest.ingestion.base import RawDocument
from fitz_ai.ingest.ingestion.registry import get_ingest_plugin


def test_local_ingest_returns_raw_documents(tmp_path):
    p = tmp_path / "b.txt"
    p.write_text("world", encoding="utf-8")

    plugin_cls = get_ingest_plugin("local")
    plugin = plugin_cls()

    docs = list(plugin.ingest(str(tmp_path), kwargs={}))
    assert docs and all(isinstance(d, RawDocument) for d in docs)
    assert docs[0].path.endswith("b.txt")
    assert docs[0].metadata["source"] == "local_fs"
