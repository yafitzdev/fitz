# fitz_ai/vector_db/__init__.py
"""
Vector DB plugin system for Fitz.

All HTTP-based vector DB plugins are YAML-driven. Drop a YAML file
into plugins/ and it works automatically. Local vector DBs (like FAISS)
use a thin YAML wrapper that points to a Python class.

Usage:
    from fitz_ai.vector_db import get_vector_db_plugin

    # HTTP-based plugins (from YAML)
    db = get_vector_db_plugin("qdrant")
    db = get_vector_db_plugin("pinecone", index_name="...", project_id="...")
    db = get_vector_db_plugin("weaviate")
    db = get_vector_db_plugin("milvus")

    # Local plugins (Python implementation)
    db = get_vector_db_plugin("local-faiss")

    # All plugins support the same interface
    db.upsert("collection", points)
    results = db.search("collection", vector, limit=10)
    count = db.count("collection")
    db.delete_collection("collection")
    collections = db.list_collections()
    stats = db.get_collection_stats("collection")
"""

from __future__ import annotations

from fitz_ai.vector_db.base import SearchResult, VectorDBPlugin
from fitz_ai.vector_db.loader import (
    GenericVectorDBPlugin,
    VectorDBSpec,
    create_vector_db_plugin,
    get_vector_db_plugin,
    load_vector_db_spec,
)

__all__ = [
    # Base types
    "SearchResult",
    "VectorDBPlugin",
    # Loader
    "VectorDBSpec",
    "GenericVectorDBPlugin",
    "load_vector_db_spec",
    "create_vector_db_plugin",
    # Main API
    "get_vector_db_plugin",
]
