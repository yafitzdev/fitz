import pytest

from fitz.llm.chat.engine import ChatEngine


class DummyChatOK:
    def chat(self, messages):
        return "ok"


class DummyChatBad:
    def chat(self, messages):
        return messages  # not a str


def test_chat_engine_returns_str():
    eng = ChatEngine(DummyChatOK())
    out = eng.chat([{"role": "user", "content": "hello"}])
    assert out == "ok"


def test_chat_engine_enforces_return_type():
    eng = ChatEngine(DummyChatBad())
    with pytest.raises(TypeError):
        eng.chat([{"role": "user", "content": "hello"}])
