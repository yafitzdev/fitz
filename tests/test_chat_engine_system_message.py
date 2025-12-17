# tests/test_chat_engine_system_message.py
from fitz.llm.chat.engine import ChatEngine


class Dummy:
    def __init__(self) -> None:
        self.seen = None

    def chat(self, messages):
        self.seen = messages
        return "ok"


def test_chat_engine_system_message():
    plugin = Dummy()
    eng = ChatEngine(plugin)

    msgs = [
        {"role": "system", "content": "system!"},
        {"role": "user", "content": "hello"},
    ]

    out = eng.chat(msgs)

    assert out == "ok"
    assert plugin.seen == msgs
