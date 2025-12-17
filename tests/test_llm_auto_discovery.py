# tests/test_llm_auto_discovery.py
"""
Tests for LLM plugin auto-discovery.

Architecture note:
- LLM plugins: chat, embedding, rerank
- Vector DB plugins have their OWN registry (fitz.vector_db.registry)
- Do NOT mix them!
"""
import pytest

from fitz.llm import available_llm_plugins, get_llm_plugin
from fitz.llm.registry import LLMRegistryError


def test_available_llm_plugins_smoke():
    """Should return a list for each LLM plugin type."""
    # These are the ONLY valid LLM plugin types
    # vector_db is NOT an LLM - it has its own registry
    llm_types = ("chat", "embedding", "rerank")

    for plugin_type in llm_types:
        result = available_llm_plugins(plugin_type)
        assert isinstance(result, list), f"Expected list for {plugin_type}"


def test_vector_db_has_separate_registry():
    """Vector DB plugins use a separate registry, not LLM registry."""
    from fitz.vector_db.registry import available_vector_db_plugins

    result = available_vector_db_plugins()
    assert isinstance(result, list)
    # Should have at least qdrant or local-faiss
    assert len(result) > 0


def test_get_llm_plugin_unknown_raises():
    """Unknown plugin should raise LLMRegistryError."""
    with pytest.raises(LLMRegistryError):
        get_llm_plugin(plugin_name="__nonexistent__", plugin_type="chat")


def test_get_llm_plugin_invalid_type_raises():
    """Invalid plugin type (like vector_db) should raise ValueError."""
    with pytest.raises(ValueError):
        get_llm_plugin(plugin_name="cohere", plugin_type="vector_db")


def test_get_llm_plugin_returns_class_if_present():
    """Registered plugins should return a class."""
    for kind in ("chat", "embedding", "rerank"):
        names = available_llm_plugins(kind)
        if "cohere" not in names:
            pytest.skip(f"'cohere' not registered for kind={kind!r}")

        cls = get_llm_plugin(plugin_name="cohere", plugin_type=kind)
        assert isinstance(cls, type)
