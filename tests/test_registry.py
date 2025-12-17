# tests/test_registry.py
"""
Tests for the centralized plugin registry system.

Architecture:
- All registries use fitz.core.registry.PluginRegistry
- PluginNotFoundError is raised for unknown plugins
- Each plugin type has its own registry instance
"""
import pytest

from fitz.core.registry import PluginNotFoundError
from fitz.ingest.ingestion.plugins.local_fs import LocalFSIngestPlugin
from fitz.ingest.ingestion.registry import get_ingest_plugin


def test_registry_returns_correct_plugin():
    """Registry should return the correct plugin class by name."""
    plugin_cls = get_ingest_plugin("local")
    assert plugin_cls is LocalFSIngestPlugin


def test_registry_rejects_unknown_plugin():
    """Registry should raise PluginNotFoundError for unknown plugins."""
    with pytest.raises(PluginNotFoundError):
        get_ingest_plugin("does_not_exist")


def test_registry_error_message_is_helpful():
    """Error message should list available plugins."""
    with pytest.raises(PluginNotFoundError) as exc_info:
        get_ingest_plugin("nonexistent_plugin")

    error_msg = str(exc_info.value)

    # Should mention the requested plugin
    assert "nonexistent_plugin" in error_msg

    # Should list available plugins
    assert "Available" in error_msg
    assert "local" in error_msg


def test_all_registries_use_same_pattern():
    """All plugin registries should follow the same pattern."""
    from fitz.core.registry import (
        PluginNotFoundError,
        get_ingest_plugin,
        get_llm_plugin,
        get_vector_db_plugin,
    )

    # All should raise PluginNotFoundError for unknown plugins
    with pytest.raises(PluginNotFoundError):
        get_llm_plugin(plugin_name="__fake__", plugin_type="chat")

    with pytest.raises(PluginNotFoundError):
        get_vector_db_plugin("__fake__")

    with pytest.raises(PluginNotFoundError):
        get_ingest_plugin("__fake__")
