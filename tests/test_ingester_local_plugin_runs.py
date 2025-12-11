import os
from fitz_ingest.ingester.engine import Ingester
from fitz_ingest.ingester.plugins.base import RawDocument

def test_ingester_local_plugin_runs(tmp_path):
    # Create a temporary folder with a test file
    file = tmp_path / "doc.txt"
    file.write_text("hello world")

    ingester = Ingester(plugin_name="local", config={})
    docs = list(ingester.run(str(tmp_path)))

    assert len(docs) == 1
    assert isinstance(docs[0], RawDocument)
    assert docs[0].content == "hello world"
