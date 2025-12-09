# src/fitz_rag/retriever/qdrant_client.py
"""
Qdrant client wrapper for fitz_rag.

This module provides a thin abstraction over the qdrant-client
constructor and convenience helpers used across the framework.

Users can configure Qdrant via:
- Unified YAML config (fitz_rag.config)
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

# NEW — unified config
from fitz_rag.config import get_config
_cfg = get_config()


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
    Create a QdrantClient using either:
    - explicit parameters
    - unified YAML config
    - environment variables
    - fallback to http://localhost:6333
    """

    qcfg = _cfg.get("qdrant", {})

    # unified config as fallback
    cfg_url = f"{'https' if qcfg.get('https', False) else 'http'}://{qcfg.get('host', 'localhost')}:{qcfg.get('port', 6333)}"

    url = (
        url
        or os.getenv(ENV_QDRANT_URL)
        or cfg_url
    )

    api_key = (
        api_key
        or os.getenv(ENV_QDRANT_API_KEY)
        or None
    )

    if timeout is None:
        # ENV > unified config > default 30
        env_val = os.getenv(ENV_QDRANT_TIMEOUT)
        if env_val:
            timeout = int(env_val)
        else:
            timeout = int(qcfg.get("timeout", 30))

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
