# tests/test_default_preset_resolves.py
"""
Test that default configuration resolves correctly.

Note: The old preset system has been removed. This test now verifies
that the unified config loads correctly.
"""

from fitz.engines.classic_rag.config import load_config, ClassicRagConfig


def test_default_preset_resolves_to_runtime_config():
    """Test that default config loads and has all required fields."""
    cfg = load_config()

    assert isinstance(cfg, ClassicRagConfig)

    # Required plugin configs
    assert cfg.llm.plugin_name
    assert cfg.embedding.plugin_name
    assert cfg.vector_db.plugin_name

    # Retriever is required
    assert cfg.retriever.plugin_name
    assert cfg.retriever.collection

    # Rerank is optional but should have a default
    assert cfg.rerank is not None