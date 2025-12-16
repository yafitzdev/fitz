# tests/test_default_preset_resolves.py
from fitz.core.config.loader import load_config
from fitz.core.config.schema import FitzConfig


def test_default_preset_resolves_to_runtime_config():
    cfg = load_config()

    assert isinstance(cfg, FitzConfig)

    assert cfg.chat.plugin_name
    assert cfg.embedding.plugin_name
    assert cfg.vector_db.plugin_name
    assert cfg.pipeline.plugin_name

    assert cfg.rerank is None or cfg.rerank.plugin_name
