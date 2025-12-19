# tests/test_meta_config_shape.py
"""
Test that default.yaml has the correct structure.

Note: The old meta config with presets has been removed.
This test now verifies the direct config structure.
"""

from fitz_ai.engines.classic_rag.config.loader import DEFAULT_CONFIG_PATH, _load_yaml


def test_default_yaml_has_correct_structure():
    """
    Verifies the default.yaml has the required top-level keys.

    The config should have:
    - chat: Chat/LLM plugin config
    - embedding: Embedding plugin config
    - vector_db: Vector database config
    - retriever: Retriever config

    Optional:
    - rerank: Reranker config
    - rgs: RGS config
    - logging: Logging config
    """
    data = _load_yaml(DEFAULT_CONFIG_PATH)

    # Required keys
    assert "chat" in data, "Missing 'chat' config"
    assert "embedding" in data, "Missing 'embedding' config"
    assert "vector_db" in data, "Missing 'vector_db' config"
    assert "retriever" in data, "Missing 'retriever' config"

    # Plugin configs should have plugin_name
    assert "plugin_name" in data["chat"]
    assert "plugin_name" in data["embedding"]
    assert "plugin_name" in data["vector_db"]
    assert "plugin_name" in data["retriever"]

    # Retriever should have collection
    assert "collection" in data["retriever"]
