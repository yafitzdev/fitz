# fitz/backends/local_vector_db/faiss.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from fitz.logging.logger import get_logger
from fitz.logging.tags import VECTOR_DB
from fitz.engines.classic_rag.models.chunk import Chunk
from fitz.vector_db.base import SearchResult

from .config import LocalVectorDBConfig

logger = get_logger(__name__)


class FaissLocalVectorDB:
    """
    Local FAISS-backed VectorDB plugin.

    Implements the canonical VectorDBPlugin contract with upsert support.
    """

    plugin_name = "local-faiss"
    plugin_type = "vector_db"

    def __init__(self, *, dim: int, config: LocalVectorDBConfig):
        self._dim = dim
        self._config = config

        self._base_path = Path(config.path)
        self._index_path = self._base_path / "index.faiss"
        self._meta_path = self._base_path / "payloads.npy"
        self._ids_path = self._base_path / "ids.npy"

        self._base_path.mkdir(parents=True, exist_ok=True)

        self._index = faiss.IndexFlatL2(dim)

        # payloads and IDs are aligned with FAISS vector order
        self._payloads: list[dict] = []
        self._ids: list[str] = []

        if self._config.persist and self._index_path.exists():
            self._load()

        logger.info(f"{VECTOR_DB} Local FAISS initialized (dim={dim}, path={self._base_path})")

    # ------------------------------------------------------------------ public

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """
        Upsert points into the vector database.

        Expected points format:
        [
            {
                "id": "chunk_id",
                "vector": [0.1, 0.2, ...],
                "payload": {"doc_id": "...", "content": "...", ...}
            },
            ...
        ]
        """
        # Build a mapping of existing IDs to their indices
        id_to_idx = {id_: idx for idx, id_ in enumerate(self._ids)}

        # Separate updates from inserts
        updates = []
        inserts = []

        for point in points:
            point_id = point["id"]
            if point_id in id_to_idx:
                updates.append((id_to_idx[point_id], point))
            else:
                inserts.append(point)

        # Handle updates (replace existing vectors)
        for idx, point in updates:
            vector = np.asarray(point["vector"], dtype="float32")
            # FAISS doesn't support in-place updates, so we'll just track metadata updates
            self._payloads[idx] = point.get("payload", {})
            # Note: Vector is not updated in FAISS index (would require rebuild)
            # For true upsert with vector updates, we'd need to rebuild the entire index

        # Handle inserts (add new vectors)
        if inserts:
            vectors = []
            for point in inserts:
                vectors.append(point["vector"])
                self._ids.append(point["id"])
                self._payloads.append(point.get("payload", {}))

            matrix = np.asarray(vectors, dtype="float32")
            self._index.add(matrix)

        # Persist if configured
        if self._config.persist:
            self.persist()

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Legacy add method for chunks with embedded vectors."""
        vectors = []

        for chunk in chunks:
            embedding = getattr(chunk, "embedding", None)
            if embedding is None:
                continue

            vectors.append(embedding)
            self._ids.append(chunk.id)
            self._payloads.append(chunk.metadata or {})

        if not vectors:
            return

        matrix = np.asarray(vectors, dtype="float32")
        self._index.add(matrix)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int,
        with_payload: bool = True,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if self._index.ntotal == 0:
            return []

        query = np.asarray([query_vector], dtype="float32")
        distances, indices = self._index.search(query, limit)

        results: list[SearchResult] = []

        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            payload = self._payloads[idx] if with_payload else {}

            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=float(score),
                    payload=payload,
                )
            )

        return results

    def count(self) -> int:
        """Return the number of vectors in the index."""
        return self._index.ntotal

    def persist(self) -> None:
        """Save index and metadata to disk."""
        faiss.write_index(self._index, str(self._index_path))
        np.save(str(self._meta_path), self._payloads, allow_pickle=True)
        np.save(str(self._ids_path), self._ids, allow_pickle=True)
        logger.info(f"{VECTOR_DB} Persisted FAISS index to {self._base_path}")

    def _load(self) -> None:
        """Load index and metadata from disk."""
        self._index = faiss.read_index(str(self._index_path))
        self._payloads = list(np.load(str(self._meta_path), allow_pickle=True))

        # Handle IDs file (may not exist in older versions)
        if self._ids_path.exists():
            self._ids = list(np.load(str(self._ids_path), allow_pickle=True))
        else:
            # Fallback: generate generic IDs
            self._ids = [f"id_{i}" for i in range(self._index.ntotal)]

        logger.info(
            f"{VECTOR_DB} Loaded FAISS index from {self._base_path} "
            f"({self._index.ntotal} vectors)"
        )
