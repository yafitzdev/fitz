# fitz/vector_db/plugins/local.py
"""
Local FAISS vector database plugin.

This module re-exports FaissLocalVectorDB for plugin discovery.
The actual implementation is in fitz.backends.local_vector_db.faiss.
"""

from fitz.backends.local_vector_db.faiss import FaissLocalVectorDB

# Re-export for plugin discovery
# The FaissLocalVectorDB class has:
#   plugin_name = "local-faiss"
#   plugin_type = "vector_db"
# So it will be discovered automatically by the registry

__all__ = ["FaissLocalVectorDB"]
