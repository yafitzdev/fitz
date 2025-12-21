# tests/test_yaml_plugin_discovery.py
"""Tests for YAML plugin discovery system."""

import pytest

from fitz_ai.llm import available_llm_plugins, get_llm_plugin, list_plugins
from fitz_ai.llm.registry import LLMRegistryError


class TestYAMLPluginDiscovery:
    """Tests for YAML-based plugin discovery."""

    def test_list_plugins(self):
        """list_plugins returns available plugins."""
        chat_plugins = list_plugins("chat")
        assert isinstance(chat_plugins, list)
        assert len(chat_plugins) > 0

    def test_available_llm_plugins(self):
        """available_llm_plugins works with YAML system."""
        chat_plugins = available_llm_plugins("chat")
        assert isinstance(chat_plugins, list)
        assert len(chat_plugins) > 0

    def test_get_llm_plugin_returns_instance(self):
        """get_llm_plugin returns instance for YAML plugins."""
        yaml_plugins = list_plugins("chat")
        if not yaml_plugins:
            pytest.skip("No YAML chat plugins")

        instance = get_llm_plugin(
            plugin_name=yaml_plugins[0],
            plugin_type="chat",
            api_key="test_key_for_testing",
        )

        assert hasattr(instance, "plugin_name")
        assert hasattr(instance, "plugin_type")
        assert instance.plugin_name == yaml_plugins[0]
        assert instance.plugin_type == "chat"

    def test_invalid_plugin_type_raises(self):
        """Invalid plugin type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid LLM plugin type"):
            get_llm_plugin(plugin_name="cohere", plugin_type="invalid_type")

    def test_unknown_plugin_raises(self):
        """Unknown plugin raises LLMRegistryError."""
        with pytest.raises(LLMRegistryError, match="Unknown chat plugin"):
            get_llm_plugin(plugin_name="__nonexistent__", plugin_type="chat")
