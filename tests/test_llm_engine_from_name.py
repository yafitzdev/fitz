# tests/test_llm_engine_from_name.py

from fitz_stack.llm import get_llm_plugin, LLMRegistryError
from fitz_stack.llm.chat import ChatEngine
from fitz_stack.llm.embedding import EmbeddingEngine
from fitz_stack.llm.rerank import RerankEngine


def test_chat_engine_from_name():
    cls = get_llm_plugin("cohere", "chat")
    eng = ChatEngine.from_name("cohere", api_key="x", model="y")
    assert isinstance(eng.plugin, cls)


def test_embedding_engine_from_name():
    cls = get_llm_plugin("cohere", "embedding")
    eng = EmbeddingEngine.from_name("cohere", api_key="x", model="y")
    assert isinstance(eng.plugin, cls)


def test_rerank_engine_from_name():
    cls = get_llm_plugin("cohere", "rerank")
    eng = RerankEngine.from_name("cohere", api_key="x", model="rerank-english-v3.0")
    assert isinstance(eng.plugin, cls)


def test_get_llm_plugin_unknown_raises():
    try:
        get_llm_plugin("nope", "chat")
        assert False, "Expected LLMRegistryError"
    except LLMRegistryError:
        pass
