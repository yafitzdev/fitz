from fitz_rag.llm.chat.engine import ChatEngine
from fitz_rag.llm.chat.base import ChatPlugin

class Dummy(ChatPlugin):
    plugin_name = "dummy"
    def chat(self, messages):
        return messages  # just echo for inspection

def test_chat_engine_system_message():
    eng = ChatEngine(Dummy())
    msgs = eng.chat_text("hello", system_prompt="system!")

    # Dummy returns the list of messages
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "system!"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "hello"
