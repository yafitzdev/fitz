# ============================
# File: tests/test_registry.py
# ============================
from ingest.ingestion.registry import get_ingest_plugin
from ingest.ingestion.plugins.local_fs import LocalFSIngestPlugin


def test_registry_returns_correct_plugin():
    plugin_cls = get_ingest_plugin("local")
    assert plugin_cls is LocalFSIngestPlugin


def test_registry_rejects_unknown_plugin():
    try:
        get_ingest_plugin("does_not_exist")
        assert False, "Expected ValueError"
    except ValueError:
        assert True
