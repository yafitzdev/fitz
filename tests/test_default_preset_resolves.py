# tests/test_default_preset_resolves.py
"""Test that default config loads and resolves correctly."""

from fitz_ai.engines.classic_rag.config import ClassicRagConfig, load_config


def test_default_preset_resolves_to_runtime_config():
    """Test that default config loads and has all required fields."""
    cfg = load_config()

    assert isinstance(cfg, ClassicRagConfig)

    # Required plugin configs - uses 'chat' not 'llm'
    assert cfg.chat.plugin_name
    assert cfg.embedding.plugin_name
    assert cfg.vector_db.plugin_name
    assert cfg.retrieval.plugin_name
    assert cfg.retrieval.collection