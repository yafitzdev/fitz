# fitz_ai/engines/classic_rag/retrieval/steps/artifact_fetch.py
"""
Artifact fetch step for retrieval pipeline.

This step fetches all artifacts from the vector DB and prepends them
to the retrieval results with score=1.0, ensuring they are always included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.engines.classic_rag.retrieval.steps.base import RetrievalStep


@runtime_checkable
class ArtifactClient(Protocol):
    """Protocol for fetching artifacts from vector DB."""

    def fetch_artifacts(self, collection: str) -> List[dict]:
        """
        Fetch all artifacts from a collection.

        Args:
            collection: Collection name

        Returns:
            List of artifact payloads
        """
        ...


@dataclass
class ArtifactFetchStep(RetrievalStep):
    """
    Fetches artifacts and prepends them to results.

    Artifacts are always included with score=1.0 to ensure they
    appear in every query result. This provides consistent context
    about the codebase structure.

    Usage:
        step = ArtifactFetchStep(
            artifact_client=client,
            collection="my_collection",
        )
        chunks = step.execute(query, existing_chunks)
    """

    artifact_client: ArtifactClient
    collection: str
    score: float = 1.0  # Score to assign to artifacts

    def execute(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """Fetch artifacts and prepend to chunk list."""
        # Fetch artifacts from vector DB
        artifact_payloads = self.artifact_client.fetch_artifacts(self.collection)

        if not artifact_payloads:
            return chunks

        # Convert payloads to Chunks
        artifact_chunks = []
        for payload in artifact_payloads:
            chunk = Chunk(
                id=payload.get("id", f"artifact:{payload.get('artifact_type', 'unknown')}"),
                doc_id=payload.get("doc_id", "artifact"),
                content=payload.get("content", ""),
                chunk_index=0,
                metadata={
                    "is_artifact": True,
                    "artifact_type": payload.get("artifact_type"),
                    "title": payload.get("title"),
                    "score": self.score,
                    "rerank_score": self.score,
                    **{k: v for k, v in payload.items()
                       if k not in ("content", "id", "doc_id", "is_artifact", "artifact_type", "title")},
                },
            )
            artifact_chunks.append(chunk)

        # Prepend artifacts to results
        return artifact_chunks + chunks


class SimpleArtifactClient:
    """
    Simple artifact client that fetches artifacts from vector DB.

    Works with both filter-capable DBs (Qdrant) and scroll-based DBs (FAISS).
    """

    def __init__(self, vector_db: Any):
        """
        Initialize with a vector DB instance.

        Args:
            vector_db: Vector DB with search or scroll method
        """
        self._vdb = vector_db

    def fetch_artifacts(self, collection: str) -> List[dict]:
        """Fetch all artifacts from the collection."""
        # Try scroll-based approach first (works with FAISS)
        if hasattr(self._vdb, "scroll"):
            return self._fetch_via_scroll(collection)

        # Fall back to filter-based search (works with Qdrant)
        return self._fetch_via_filter(collection)

    def _fetch_via_scroll(self, collection: str) -> List[dict]:
        """Fetch artifacts by scrolling through all records and filtering."""
        try:
            payloads = []
            offset = 0
            batch_size = 100

            while True:
                records, next_offset = self._vdb.scroll(
                    collection=collection,
                    limit=batch_size,
                    offset=offset,
                )

                for record in records:
                    payload = record.payload if hasattr(record, "payload") else {}
                    if payload.get("is_artifact"):
                        payloads.append(payload)

                if next_offset is None:
                    break
                offset = next_offset

            return payloads

        except Exception:
            return []

    def _fetch_via_filter(self, collection: str) -> List[dict]:
        """Fetch artifacts using filter-based search (for Qdrant-like DBs)."""
        try:
            results = self._vdb.search(
                collection_name=collection,
                query_vector=[0.0] * 1024,  # Dummy vector
                limit=100,
                with_payload=True,
                filter_={
                    "must": [
                        {"key": "is_artifact", "match": {"value": True}}
                    ]
                }
            )

            payloads = []
            for result in results:
                if hasattr(result, "payload"):
                    payloads.append(result.payload)
                elif isinstance(result, dict):
                    payloads.append(result.get("payload", result))

            return payloads

        except Exception:
            return []


__all__ = ["ArtifactFetchStep", "ArtifactClient", "SimpleArtifactClient"]
