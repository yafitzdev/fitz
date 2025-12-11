"""
Built-in embedding plugins for fitz-rag.

Each plugin module should define:
    - a class with:
        * class attribute `plugin_name: str` (unique)
        * method `embed(self, text: str) -> List[float]`

Plugins are auto-discovered by:
    fitz_rag.llm.embedding.registry.auto_discover_plugins()
"""
