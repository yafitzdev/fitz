from fitz_ingest.ingester.registry import REGISTRY, get_ingest_plugin
from fitz_ingest.ingester.plugins.local_fs import LocalFSIngestPlugin

def test_registry_returns_correct_plugin():
    plugin = get_ingest_plugin("local")
    assert isinstance(plugin, LocalFSIngestPlugin)

def test_registry_rejects_unknown_plugin():
    try:
        get_ingest_plugin("does_not_exist")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
