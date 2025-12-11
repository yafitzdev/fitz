"""
Built-in retrieval plugins for fitz-rag.

Each plugin module should define:
    - a class implementing `retrieve(self, query: str) -> List[Chunk]`
    - a class attribute `plugin_name: str` (unique)
"""
