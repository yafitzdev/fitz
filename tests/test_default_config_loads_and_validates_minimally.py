# tests/test_default_config_loads_and_validates_minimally.py
"""
Test that the default config can be loaded and validated.
"""

from fitz_ai.engines.classic_rag.config import ClassicRagConfig, load_config


def test_default_config_loads_and_validates_base_schema():
    """
    Verifies the CURRENT architectural contract:

    - default.yaml can be loaded
    - resolved runtime config validates against ClassicRagConfig
    - all required fields are present
    """

    cfg = load_config()

    assert isinstance(cfg, ClassicRagConfig)

    # Core plugin configs - uses 'chat' not 'llm'
    assert cfg.chat.plugin_name
    assert cfg.embedding.plugin_name
    assert cfg.vector_db.plugin_name

    # Retrieval (YAML plugin reference)
    assert cfg.retrieval.plugin_name
    assert cfg.retrieval.collection

    # RGS config exists
    assert cfg.rgs is not None
