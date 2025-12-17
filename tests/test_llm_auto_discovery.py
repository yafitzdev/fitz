import pytest

from fitz.llm import available_llm_plugins, get_llm_plugin
from fitz.llm.registry import LLMRegistryError


def test_available_llm_plugins_smoke():
    # Should not crash and should return a list.
    kinds = ("chat", "embedding", "rerank", "vector_db")
    for k in kinds:
        assert isinstance(available_llm_plugins(k), list)


def test_get_llm_plugin_unknown_raises():
    with pytest.raises(LLMRegistryError):
        get_llm_plugin(plugin_name="__nope__", plugin_type="chat")


def test_get_llm_plugin_returns_class_if_present():
    # If your package ships "cohere" plugins, assert they exist.
    # If not, this test will skip instead of forcing a dependency.
    for kind in ("chat", "embedding", "rerank"):
        names = available_llm_plugins(kind)
        if "cohere" not in names:
            pytest.skip(f"'cohere' not registered for kind={kind!r}")
        cls = get_llm_plugin(plugin_name="cohere", plugin_type=kind)
        assert isinstance(cls, type)
