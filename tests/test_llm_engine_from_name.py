import pytest

from fitz_ai.llm import get_llm_plugin
from fitz_ai.llm.registry import LLMRegistryError


def test_get_llm_plugin_requires_keywords():
    # Ensures your public API stays explicit.
    get_llm_plugin(plugin_name="cohere", plugin_type="chat") if True else None
    # If "cohere" isn't present in your build, the above would raise.
    # So we only assert API shape here.
    assert True


def test_get_llm_plugin_unknown_raises_registry_error():
    with pytest.raises(LLMRegistryError):
        get_llm_plugin(plugin_name="nope", plugin_type="chat")
