# fitz_ai/backends/local_vector_db/faiss.py
"""
Local FAISS-backed Vector Database.

Design principles:
- Lazy initialization: dimension detected on first upsert
- Uses FitzPaths for storage location
- No special handling needed vs other vector DBs
- Implements standard VectorDBPlugin contract
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import VECTOR_DB
from fitz_ai.vector_db.base import SearchResult

logger = get_logger(__name__)


@dataclass
class _FaissRecord:
    """Simple record object for FAISS scroll results."""

    id: str
    payload: Dict[str, Any]


class FaissLocalVectorDB:
    """
    Local FAISS-backed VectorDB plugin.

    Key design: Lazy initialization.
    - Dimension is detected automatically on first upsert
    - No need to specify dim at construction time
    - This makes it work like other vector DBs (Qdrant, etc.)

    Usage:
        # Simple init - just like Qdrant
        db = FaissLocalVectorDB()

        # Or with custom path
        db = FaissLocalVectorDB(path=".fitz_ai/vector_db")

        # Dimension is auto-detected on first upsert
        db.upsert("collection", [{"id": "1", "vector": [0.1, 0.2, ...], "payload": {...}}])
    """

    plugin_name = "local-faiss"
    plugin_type = "vector_db"

    def __init__(
        self,
        *,
        path: Optional[str | Path] = None,
        persist: bool = True,
    ):
        """
        Initialize FAISS vector database.

        Args:
            path: Storage path. If None, uses FitzPaths.vector_db()
            persist: Whether to persist index to disk (default: True)
        """
        # Import here to allow graceful failure if faiss not installed
        try:
            import faiss

            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss is required for local-faiss plugin. "
                "Install with: pip install faiss-cpu"
            )

        # Resolve path
        if path is None:
            from fitz_ai.core.paths import FitzPaths

            self._base_path = FitzPaths.vector_db()
        else:
            self._base_path = Path(path)

        self._persist = persist

        # Lazy initialization - these are set on first upsert
        self._dim: Optional[int] = None
        self._index: Optional[Any] = None  # faiss.IndexFlatL2
        self._payloads: List[Dict] = []
        self._ids: List[str] = []

        # File paths
        self._index_path = self._base_path / "index.faiss"
        self._meta_path = self._base_path / "payloads.npy"
        self._ids_path = self._base_path / "ids.npy"
        self._dim_path = self._base_path / "dim.txt"

        # Try to load existing index
        self._try_load()

        logger.info(
            f"{VECTOR_DB} Local FAISS initialized "
            f"(path={self._base_path}, dim={self._dim or 'auto'})"
        )

    def _try_load(self) -> bool:
        """Try to load existing index from disk. Returns True if successful."""
        if not self._index_path.exists():
            return False

        try:
            # Load dimension
            if self._dim_path.exists():
                self._dim = int(self._dim_path.read_text().strip())

            # Load index
            self._index = self._faiss.read_index(str(self._index_path))

            # Infer dimension from index if not stored
            if self._dim is None and self._index.ntotal > 0:
                self._dim = self._index.d

            # Load metadata
            if self._meta_path.exists():
                self._payloads = list(np.load(str(self._meta_path), allow_pickle=True))

            # Load IDs
            if self._ids_path.exists():
                self._ids = list(np.load(str(self._ids_path), allow_pickle=True))
            else:
                # Fallback: generate generic IDs
                self._ids = [f"id_{i}" for i in range(self._index.ntotal)]

            logger.info(
                f"{VECTOR_DB} Loaded FAISS index from {self._base_path} "
                f"({self._index.ntotal} vectors, dim={self._dim})"
            )
            return True

        except Exception as e:
            logger.warning(f"{VECTOR_DB} Failed to load FAISS index: {e}")
            self._index = None
            self._dim = None
            self._payloads = []
            self._ids = []
            return False

    def _ensure_initialized(self, dim: int) -> None:
        """Initialize index with given dimension if not already initialized."""
        if self._index is not None:
            # Verify dimension matches
            if self._dim != dim:
                raise ValueError(
                    f"Dimension mismatch: index has dim={self._dim}, "
                    f"but got vectors with dim={dim}. "
                    f"Delete {self._base_path} to reset."
                )
            return

        # Initialize new index
        self._dim = dim
        self._index = self._faiss.IndexFlatL2(dim)
        self._payloads = []
        self._ids = []

        # Ensure directory exists
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Save dimension for future loads
        self._dim_path.write_text(str(dim))

        logger.info(f"{VECTOR_DB} Initialized new FAISS index (dim={dim})")

    def _save(self) -> None:
        """Save index and metadata to disk."""
        if self._index is None:
            return

        self._base_path.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self._index, str(self._index_path))
        np.save(str(self._meta_path), self._payloads, allow_pickle=True)
        np.save(str(self._ids_path), self._ids, allow_pickle=True)
        self._dim_path.write_text(str(self._dim))

        logger.debug(f"{VECTOR_DB} Persisted FAISS index to {self._base_path}")

    # =========================================================================
    # Public API - Standard VectorDBPlugin Contract
    # =========================================================================

    def upsert(
        self,
        collection: str,
        points: List[Dict[str, Any]],
        defer_persist: bool = False,
    ) -> None:
        """
        Upsert points into the vector database.

        Note: FAISS doesn't support true collections, so collection name is
        stored in metadata but all vectors go in one index.

        Args:
            collection: Collection name (stored in metadata)
            points: List of points with 'id', 'vector', and 'payload' keys
            defer_persist: If True, don't persist to disk (call flush() later)
        """
        if not points:
            return

        # Auto-detect dimension from first vector
        first_vector = points[0]["vector"]
        dim = len(first_vector)
        self._ensure_initialized(dim)

        for point in points:
            point_id = str(point["id"])
            vector = np.asarray([point["vector"]], dtype="float32")
            payload = point.get("payload", {})

            # Add collection to payload for filtering
            payload["_collection"] = collection

            # Check if ID exists (for update)
            if point_id in self._ids:
                idx = self._ids.index(point_id)
                # FAISS doesn't support in-place vector updates
                # Just update payload
                self._payloads[idx] = payload
            else:
                # Add new point
                self._index.add(vector)
                self._ids.append(point_id)
                self._payloads.append(payload)

        # Auto-persist if enabled and not deferred
        if self._persist and not defer_persist:
            self._save()

    def flush(self) -> None:
        """Explicitly persist index to disk."""
        if self._persist:
            self._save()

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        with_payload: bool = True,
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            collection_name: Collection to search (filters results)
            query_vector: Query vector
            limit: Maximum number of results
            with_payload: Whether to include payload in results

        Returns:
            List of SearchResult objects
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Search more than limit to account for collection filtering
        search_limit = min(limit * 3, self._index.ntotal)

        query = np.asarray([query_vector], dtype="float32")
        distances, indices = self._index.search(query, search_limit)

        results: List[SearchResult] = []

        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            payload = self._payloads[idx] if idx < len(self._payloads) else {}

            # Filter by collection
            if payload.get("_collection") != collection_name:
                continue

            # Convert L2 distance to similarity score (lower distance = higher score)
            # Using 1 / (1 + distance) to get a 0-1 score
            score = 1.0 / (1.0 + float(distance))

            result_payload = dict(payload) if with_payload else {}
            # Remove internal fields from payload
            result_payload.pop("_collection", None)

            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=score,
                    payload=result_payload,
                )
            )

            if len(results) >= limit:
                break

        return results

    def list_collections(self) -> List[str]:
        """
        List all collections stored in this FAISS index.

        FAISS doesn't have native collections - we store collection name
        in payload._collection field. This extracts unique collection names.

        Returns:
            Sorted list of collection names
        """
        if not self._payloads:
            return []

        collections = set()
        for payload in self._payloads:
            coll = payload.get("_collection")
            if coll:
                collections.add(coll)

        return sorted(collections)

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.

        Args:
            collection: Collection name

        Returns:
            Dict with points_count, vectors_count, status, vector_size
        """
        count = sum(1 for p in self._payloads if p.get("_collection") == collection)

        return {
            "points_count": count,
            "vectors_count": count,
            "status": "ready",
            "vector_size": self._dim,
        }

    def scroll(
        self,
        collection: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[_FaissRecord], Optional[int]]:
        """
        Scroll through records in a collection.

        Args:
            collection: Collection name
            limit: Max records to return
            offset: Starting position

        Returns:
            Tuple of (records, next_offset or None if done)
        """
        # Filter to collection
        matching = []
        for i, payload in enumerate(self._payloads):
            if payload.get("_collection") == collection:
                record_id = self._ids[i] if i < len(self._ids) else f"id_{i}"
                # Remove internal fields from payload
                clean_payload = {k: v for k, v in payload.items() if k != "_collection"}
                matching.append(_FaissRecord(record_id, clean_payload))

        # Apply offset and limit
        start = offset
        end = offset + limit
        batch = matching[start:end]

        next_offset = end if end < len(matching) else None

        return batch, next_offset

    def scroll_with_vectors(
        self,
        collection: str,
        limit: int = 10,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Scroll through records in a collection, including vectors.

        Used by knowledge map to fetch embeddings for visualization.

        Args:
            collection: Collection name
            limit: Max records to return
            offset: Starting position

        Returns:
            Tuple of (records_with_vectors, next_offset or None if done)
            Each record is dict with 'id', 'payload', 'vector'
        """
        if self._index is None:
            return [], None

        # Filter to collection and include vectors
        matching = []
        for i, payload in enumerate(self._payloads):
            if payload.get("_collection") == collection:
                record_id = self._ids[i] if i < len(self._ids) else f"id_{i}"
                clean_payload = {k: v for k, v in payload.items() if k != "_collection"}

                # Reconstruct vector from FAISS index
                vector = self._index.reconstruct(i).tolist()

                matching.append(
                    {
                        "id": record_id,
                        "payload": clean_payload,
                        "vector": vector,
                    }
                )

        # Apply offset and limit
        start = offset
        end = offset + limit
        batch = matching[start:end]

        next_offset = end if end < len(matching) else None

        return batch, next_offset

    def count(self, collection: Optional[str] = None) -> int:
        """
        Return the number of vectors.

        Args:
            collection: If provided, count only vectors in this collection
        """
        if self._index is None:
            return 0

        if collection is None:
            return self._index.ntotal

        # Count by collection
        return sum(1 for p in self._payloads if p.get("_collection") == collection)

    def delete_collection(self, collection: str) -> int:
        """
        Delete all vectors in a collection.

        Note: FAISS doesn't support efficient deletion, so this rebuilds the index.

        Returns:
            Number of vectors deleted
        """
        if self._index is None:
            return 0

        # Find indices to keep
        keep_indices = [
            i
            for i, p in enumerate(self._payloads)
            if p.get("_collection") != collection
        ]

        deleted = self._index.ntotal - len(keep_indices)

        if deleted == 0:
            return 0

        if len(keep_indices) == 0:
            # Delete everything
            self._index = self._faiss.IndexFlatL2(self._dim)
            self._payloads = []
            self._ids = []
        else:
            # Rebuild index with remaining vectors
            # This is expensive but FAISS doesn't support deletion
            old_index = self._index
            self._index = self._faiss.IndexFlatL2(self._dim)

            new_payloads = []
            new_ids = []

            for i in keep_indices:
                vec = old_index.reconstruct(i).reshape(1, -1)
                self._index.add(vec)
                new_payloads.append(self._payloads[i])
                new_ids.append(self._ids[i])

            self._payloads = new_payloads
            self._ids = new_ids

        if self._persist:
            self._save()

        logger.info(
            f"{VECTOR_DB} Deleted {deleted} vectors from collection '{collection}'"
        )
        return deleted


__all__ = ["FaissLocalVectorDB"]
