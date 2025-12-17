# tests/test_default_config_loads_and_validates_minimally.py
"""
Test that default configuration loads and validates correctly.
"""

from fitz.engines.classic_rag.config import load_config, ClassicRagConfig


def test_default_config_loads_and_validates_base_schema():
    """
    Verifies the CURRENT architectural contract:

    - default.yaml can be loaded
    - resolved runtime config validates against ClassicRagConfig
    - all required fields are present
    """

    cfg = load_config()

    assert isinstance(cfg, ClassicRagConfig)

    # Core plugin configs
    assert cfg.llm.plugin_name
    assert cfg.embedding.plugin_name
    assert cfg.vector_db.plugin_name

    # Retriever config
    assert cfg.retriever.plugin_name
    assert cfg.retriever.collection
    assert cfg.retriever.top_k > 0

    # Optional configs have defaults
    assert cfg.rgs is not None
    assert cfg.logging is not None

    # Rerank is optional
    assert cfg.rerank is not None  # Has default (disabled)