from fitz_ingest.ingester.engine import Ingester
from fitz_ingest.ingester.plugins.base import RawDocument

def test_ingester_returns_raw_documents(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("sample")

    ingester = Ingester(plugin_name="local", config={})
    docs = list(ingester.run(str(tmp_path)))

    doc = docs[0]

    assert isinstance(doc, RawDocument)
    assert doc.path.endswith("sample.txt")
    assert doc.metadata["source"] == "local_fs"
