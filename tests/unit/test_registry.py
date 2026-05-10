# tests/test_registry.py
"""
Tests for the centralized plugin registry system.

Architecture:
- Python-based registries use fitz_sage.core.registry.PluginRegistry
- LLM providers use fitz_sage.llm (direct provider wrappers)
- Vector DB plugins use fitz_sage.vector_db.registry (YAML-based)
- Chunking plugins use fitz_sage.core.registry.CHUNKING_REGISTRY
"""

import pytest

from fitz_sage.core.registry import PluginNotFoundError, get_chunking_plugin


def test_chunking_registry_returns_correct_plugin():
    """Chunking registry should return the correct plugin class by name."""
    plugin_cls = get_chunking_plugin("simple")
    assert plugin_cls.plugin_name == "simple"


def test_chunking_registry_rejects_unknown_plugin():
    """Chunking registry should raise PluginNotFoundError for unknown plugins."""
    with pytest.raises(PluginNotFoundError):
        get_chunking_plugin("does_not_exist")


def test_registry_error_message_is_helpful():
    """Error message should list available plugins."""
    with pytest.raises(PluginNotFoundError) as exc_info:
        get_chunking_plugin("nonexistent_plugin")

    error_msg = str(exc_info.value)
    assert "nonexistent_plugin" in error_msg
    assert "Available" in error_msg


def test_chunking_registry_raises_for_unknown_plugin():
    """Chunking registry raises PluginNotFoundError for unknown plugins."""
    with pytest.raises(PluginNotFoundError):
        get_chunking_plugin("__fake__")


def test_llm_providers_raise_on_unknown():
    """LLM config should raise ValueError for unknown providers."""
    from fitz_sage.llm.config import create_chat_provider

    with pytest.raises(ValueError) as exc_info:
        create_chat_provider("__fake__")

    assert "Unknown chat provider" in str(exc_info.value)
