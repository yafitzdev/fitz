# tests/test_default_config_loads_and_validates_minimally.py

from fitz.engines.classic_rag.config.loader import load_config
from fitz.engines.classic_rag.config.schema import FitzConfig


def test_default_config_loads_and_validates_base_schema():
    """
    Verifies the CURRENT architectural contract:

    - default.yaml can be loaded
    - default preset is resolved
    - resolved runtime config validates against FitzConfig
    """

    cfg = load_config()

    assert isinstance(cfg, FitzConfig)

    # Explicit runtime surface (Option B)
    assert cfg.chat.plugin_name
    assert cfg.embedding.plugin_name
    assert cfg.vector_db.plugin_name
    assert cfg.pipeline.plugin_name

    # rerank is optional but explicit
    assert cfg.rerank is None or cfg.rerank.plugin_name
