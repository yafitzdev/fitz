from fitz_ai.ingestion.reader.registry import get_ingest_plugin


def test_local_ingest_plugin_runs_on_directory(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello", encoding="utf-8")

    plugin_cls = get_ingest_plugin("local")
    plugin = plugin_cls()

    out = list(plugin.ingest(str(tmp_path), kwargs={}))
    assert len(out) == 1
    assert out[0].content == "hello"
