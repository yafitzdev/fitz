# tests/test_llm_auto_discovery.py

from fitz_stack.llm.registry import (
    auto_discover_llm_plugins,
    get_llm_plugin,
)


def test_auto_discovery_finds_builtin_plugins():
    # Force re-discovery to ensure test isolation
    auto_discover_llm_plugins()

    # These plugins must exist because Cohere plugins are built-in.
    emb = get_llm_plugin("cohere", "embedding")
    rer = get_llm_plugin("cohere", "rerank")
    chat = get_llm_plugin("cohere", "chat")

    assert emb.__name__.lower().startswith("cohere")
    assert rer.__name__.lower().startswith("cohere")
    assert chat.__name__.lower().startswith("cohere")
