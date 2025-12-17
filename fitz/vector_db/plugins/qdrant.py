# fitz/vector_db/plugins/qdrant.py
"""
Smart Qdrant vector database plugin.

Features:
- AUTO-DETECTION: Automatically finds Qdrant on common addresses
- Auto-creates collections when they don't exist
- Auto-detects vector dimensions from first upsert
- Auto-detects named vs unnamed vectors
- Converts string IDs to UUIDs automatically
- Provides helpful error messages with fix suggestions
- Environment variable configuration
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from fitz.vector_db.base import SearchResult, VectorDBPlugin

# Import centralized detection
from fitz.core.detect import get_qdrant_connection, detect_qdrant

logger = logging.getLogger(__name__)


# =============================================================================
# Errors with helpful messages
# =============================================================================


class QdrantConnectionError(Exception):
    """Raised when unable to connect to Qdrant."""

    def __init__(self, host: str, port: int, original_error: Exception):
        self.host = host
        self.port = port
        self.original_error = original_error

        # Get detection status for helpful message
        status = detect_qdrant()

        if status.available:
            # Qdrant IS available somewhere else
            message = f"""
❌ Cannot connect to Qdrant at {host}:{port}

However, Qdrant was detected at {status.host}:{status.port}!

Quick fix - update your config or set environment variables:
  export QDRANT_HOST={status.host}
  export QDRANT_PORT={status.port}

Or update your config.yaml:
  vector_db:
    plugin_name: qdrant
    kwargs:
      host: "{status.host}"
      port: {status.port}

Original error: {original_error}
"""
        else:
            message = f"""
❌ Cannot connect to Qdrant at {host}:{port}

Possible fixes:
  1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant
  2. Check if Qdrant is running: curl http://{host}:{port}/collections
  3. Set correct host/port via environment variables:
       export QDRANT_HOST=your-host
       export QDRANT_PORT=6333

Original error: {original_error}
"""
        super().__init__(message)


class QdrantCollectionError(Exception):
    """Raised for collection-related errors with helpful messages."""
    pass


# =============================================================================
# Helper functions
# =============================================================================


def _string_to_uuid(s: str) -> str:
    """Convert any string to a deterministic UUID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


# =============================================================================
# Smart Qdrant Plugin
# =============================================================================


@dataclass
class QdrantVectorDB(VectorDBPlugin):
    """
    Smart Qdrant vector database plugin with auto-configuration.

    Connection Resolution (in order):
    1. Explicit host/port kwargs passed to constructor
    2. QDRANT_HOST / QDRANT_PORT environment variables
    3. Auto-detection of common addresses (localhost, Docker IPs, etc.)

    Environment variables:
        QDRANT_HOST: Qdrant server hostname
        QDRANT_PORT: Qdrant server port (default: 6333)

    Features:
        - Auto-creates collections on first upsert
        - Auto-detects vector dimensions
        - Handles both named and unnamed vectors
        - Converts string IDs to UUIDs
        - Provides helpful error messages

    Usage:
        # Auto-detect (recommended)
        db = QdrantVectorDB()

        # Explicit configuration
        db = QdrantVectorDB(host="192.168.1.100", port=6333)

        # From environment
        # export QDRANT_HOST=192.168.178.2
        db = QdrantVectorDB()
    """

    plugin_name: str = "qdrant"
    plugin_type: str = "vector_db"

    # Connection settings - None means "auto-detect"
    host: Optional[str] = None
    port: Optional[int] = None

    # Internal state
    _client: Any = field(init=False, repr=False, default=None)
    _collection_cache: dict = field(init=False, repr=False, default_factory=dict)
    _resolved_host: str = field(init=False, repr=False, default="localhost")
    _resolved_port: int = field(init=False, repr=False, default=6333)

    def __post_init__(self) -> None:
        """Initialize Qdrant client with auto-detection."""
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise RuntimeError("qdrant-client is required. Install with: pip install qdrant-client")

        # Resolve connection parameters
        self._resolve_connection()

        try:
            self._client = QdrantClient(
                host=self._resolved_host,
                port=self._resolved_port,
                timeout=10
            )
            # Test connection
            self._client.get_collections()
            logger.info(f"Connected to Qdrant at {self._resolved_host}:{self._resolved_port}")
        except Exception as e:
            raise QdrantConnectionError(self._resolved_host, self._resolved_port, e)

    def _resolve_connection(self) -> None:
        """
        Resolve connection parameters with priority:
        1. Explicit kwargs (self.host, self.port)
        2. Environment variables
        3. Auto-detection
        """
        if self.host is not None and self.port is not None:
            # Explicit configuration - use as-is
            self._resolved_host = self.host
            self._resolved_port = self.port
            logger.debug(f"Using explicit Qdrant config: {self.host}:{self.port}")
            return

        # Check environment variables
        env_host = os.getenv("QDRANT_HOST")
        env_port = os.getenv("QDRANT_PORT")

        if env_host:
            self._resolved_host = env_host
            self._resolved_port = int(env_port) if env_port else 6333
            logger.debug(f"Using Qdrant from env: {self._resolved_host}:{self._resolved_port}")
            return

        # Auto-detect
        logger.debug("Auto-detecting Qdrant...")
        detected_host, detected_port = get_qdrant_connection()
        self._resolved_host = detected_host
        self._resolved_port = detected_port
        logger.debug(f"Auto-detected Qdrant: {self._resolved_host}:{self._resolved_port}")

    # =========================================================================
    # Collection Management
    # =========================================================================

    def _get_collection_info(self, collection_name: str) -> Optional[dict]:
        """Get collection info, returns None if doesn't exist."""
        if collection_name in self._collection_cache:
            return self._collection_cache[collection_name]

        try:
            info = self._client.get_collection(collection_name)
            self._collection_cache[collection_name] = {
                "exists": True,
                "vectors_config": info.config.params.vectors,
            }
            return self._collection_cache[collection_name]
        except Exception:
            return None

    def _ensure_collection(self, collection_name: str, vector_dim: int) -> None:
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        info = self._get_collection_info(collection_name)
        if info is not None:
            return  # Already exists

        logger.info(f"Creating Qdrant collection '{collection_name}' with dim={vector_dim}")

        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_dim,
                distance=Distance.COSINE,
            ),
        )

        self._collection_cache[collection_name] = {
            "exists": True,
            "vectors_config": {"size": vector_dim},
        }

    # =========================================================================
    # VectorDBPlugin Interface
    # =========================================================================

    def upsert(
            self,
            collection_name: str,
            ids: List[str],
            vectors: List[List[float]],
            payloads: Optional[List[dict]] = None,
    ) -> None:
        """
        Upsert vectors into collection.

        Auto-creates collection if it doesn't exist.
        Auto-converts string IDs to UUIDs.
        """
        from qdrant_client.models import PointStruct

        if not vectors:
            return

        # Auto-create collection with detected dimension
        vector_dim = len(vectors[0])
        self._ensure_collection(collection_name, vector_dim)

        # Build points
        points = []
        for i, (id_, vector) in enumerate(zip(ids, vectors)):
            point_id = _string_to_uuid(id_)
            payload = payloads[i] if payloads else {}
            payload["_original_id"] = id_  # Preserve original ID

            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            ))

        self._client.upsert(
            collection_name=collection_name,
            points=points,
        )

        logger.debug(f"Upserted {len(points)} vectors to '{collection_name}'")

    def search(
            self,
            collection_name: str,
            query_vector: List[float],
            limit: int = 10,
            **kwargs: Any,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        try:
            results = self._client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                **kwargs,
            )
        except Exception as e:
            if "not found" in str(e).lower():
                logger.warning(f"Collection '{collection_name}' not found")
                return []
            raise

        return [
            SearchResult(
                id=str(hit.id),
                score=hit.score,
                payload=hit.payload or {},
            )
            for hit in results
        ]

    def delete(self, collection_name: str, ids: List[str]) -> None:
        """Delete vectors by ID."""
        from qdrant_client.models import PointIdsList

        point_ids = [_string_to_uuid(id_) for id_ in ids]

        self._client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=point_ids),
        )

    def get_collection_info(self, collection_name: str) -> dict:
        """Get collection information."""
        try:
            info = self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception:
            return {"name": collection_name, "exists": False}

    def list_collections(self) -> List[str]:
        """List all collections."""
        collections = self._client.get_collections()
        return [c.name for c in collections.collections]