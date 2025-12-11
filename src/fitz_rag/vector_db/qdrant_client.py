"""
Qdrant client utilities for fitz_rag.

This version is *architecture correct*:

- No implicit config loading (pipeline owns config)
- No legacy qdrant: {host,port} structures
- No environment overrides
- Only provides two thin helpers:
      create_qdrant_client()
      ensure_collection()

Anything more complex belongs in pipeline construction.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from fitz_rag.exceptions.retriever import VectorSearchError
from fitz_rag.exceptions.config import ConfigError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import VECTOR_DB

logger = get_logger(__name__)


# -------------------------------------------------------------------
# Client Constructor
# -------------------------------------------------------------------
def create_qdrant_client(
    host: str,
    port: int,
    https: bool = False,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> QdrantClient:
    """
    Minimal, explicit Qdrant client factory.

    The pipeline passes values from retriever_cfg:
        host = cfg.retriever.qdrant_host
        port = cfg.retriever.qdrant_port

    This module no longer loads config itself.
    """

    scheme = "https" if https else "http"
    url = f"{scheme}://{host}:{port}"

    logger.debug(f"{VECTOR_DB} Initializing Qdrant client (url={url}, timeout={timeout})")

    try:
        return QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed initializing Qdrant client: {e}")
        raise VectorSearchError(f"Failed initializing Qdrant client for URL: {url}") from e


# -------------------------------------------------------------------
# Collection Helper
# -------------------------------------------------------------------
def ensure_collection(
    client: QdrantClient,
    name: str,
    vector_size: int,
    distance: qmodels.Distance = qmodels.Distance.COSINE,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Ensure the Qdrant collection exists.

    Used by ingestion engine.
    """

    logger.debug(f"{VECTOR_DB} Ensuring collection '{name}' exists…")

    # -------- Fetch collections --------
    try:
        collections = client.get_collections().collections
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed fetching collections: {e}")
        raise VectorSearchError("Failed fetching collections from Qdrant") from e

    try:
        existing = {c.name for c in collections}
    except Exception as e:
        logger.error(f"{VECTOR_DB} Invalid collections structure: {e}")
        raise VectorSearchError("Invalid Qdrant collections response") from e

    if name in existing:
        logger.debug(f"{VECTOR_DB} Collection '{name}' already exists")
        return

    # -------- Create collection --------
    logger.info(f"{VECTOR_DB} Creating new Qdrant collection '{name}'")

    try:
        client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=distance,
            ),
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"{VECTOR_DB} Failed creating collection '{name}': {e}")
        raise VectorSearchError(f"Failed creating Qdrant collection '{name}'") from e
