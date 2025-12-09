# src/fitz_rag/retriever/qdrant_client.py
"""
Qdrant client wrapper for fitz_rag.

This module provides a thin abstraction over the qdrant-client
constructor and convenience helpers used across the framework.

Users can configure Qdrant via:
- Environment variables
- Direct initialization
- Defaults (localhost)

This keeps the rest of the library independent from raw Qdrant config.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# ---------------------------------------------------------
# Default environment variable names
# ---------------------------------------------------------

ENV_QDRANT_URL = "FITZ_RAG_QDRANT_URL"
ENV_QDRANT_API_KEY = "FITZ_RAG_QDRANT_API_KEY"
ENV_QDRANT_TIMEOUT = "FITZ_RAG_QDRANT_TIMEOUT"


# ---------------------------------------------------------
# Create a Qdrant client (shared entrypoint)
# ---------------------------------------------------------

def create_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None,
) -> QdrantClient:
    """
    Create a QdrantClient using either explicit parameters or environment
    variables. Falls back to http://localhost:6333 if nothing is provided.
    """
    url = url or os.getenv(ENV_QDRANT_URL, "http://localhost:6333")
    api_key = api_key or os.getenv(ENV_QDRANT_API_KEY, None)

    # Ensure timeout is numeric
    if timeout is None:
        env_val = os.getenv(ENV_QDRANT_TIMEOUT)
        timeout = int(env_val) if env_val else 30

    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
    )


# ---------------------------------------------------------
# Collection helper
# ---------------------------------------------------------

def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Ensure a Qdrant collection exists. If not, it is created.
    """
    collections = client.get_collections().collections
    names = {c.name for c in collections}

    if name in names:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(
            size=vector_size,
            distance=distance,
        ),
        metadata=metadata,
    )
