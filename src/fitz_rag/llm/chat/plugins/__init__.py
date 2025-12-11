"""
Built-in chat plugins for fitz-rag.

Each plugin must define:
    - class attribute plugin_name: str
    - method chat(self, messages: List[dict]) -> str

Automatic discovery happens via:
    fitz_rag.llm.chat.registry.auto_discover_plugins()
"""
