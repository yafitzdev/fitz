"""
Built-in reranking plugins for fitz-rag.

Each plugin must define:
    - class attribute plugin_name: str
    - method rerank(self, query, chunks)

Automatic discovery happens via:
    fitz_rag.llm.rerank.registry.auto_discover_plugins()
"""
