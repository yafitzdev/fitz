# fitz_ai/map/models.py
"""
Pydantic models for knowledge map state.

The knowledge_map.json file caches embeddings (NOT projections) for incremental updates.
UMAP runs fresh each time since 2D layout is global.

Embeddings are stored as float16 for ~50% space reduction with minimal accuracy loss.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChunkEmbedding(BaseModel):
    """
    Cached embedding for a single chunk.

    Stores embedding as float16 for space efficiency (~50% reduction).
    """

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(..., description="Chunk ID (matches vector DB)")
    doc_id: str = Field(..., description="Parent document ID")
    label: str = Field(..., description="Display label (truncated content or title)")
    embedding: List[float] = Field(
        ..., description="Embedding vector (stored as float16)"
    )

    # Metadata for visualization
    chunk_index: int = Field(default=0, description="Position in document")
    content_preview: str = Field(default="", description="First 200 chars of content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    # Computed fields (populated during projection, not stored)
    x: Optional[float] = Field(default=None, description="UMAP x coordinate")
    y: Optional[float] = Field(default=None, description="UMAP y coordinate")
    cluster_id: Optional[int] = Field(default=None, description="Cluster assignment")
    is_gap: bool = Field(default=False, description="Whether this is in a gap region")


class DocumentNode(BaseModel):
    """
    Document-level node for hierarchy visualization.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(..., description="Document ID")
    label: str = Field(..., description="Document filename or title")
    chunk_ids: List[str] = Field(default_factory=list, description="Child chunk IDs")

    # Computed during projection (centroid of child chunks)
    x: Optional[float] = Field(default=None)
    y: Optional[float] = Field(default=None)


class ClusterInfo(BaseModel):
    """
    Information about a detected cluster.
    """

    model_config = ConfigDict(extra="forbid")

    cluster_id: int
    label: str = Field(default="", description="Auto-generated or user label")
    chunk_count: int = Field(default=0)
    centroid_x: float = Field(default=0.0)
    centroid_y: float = Field(default=0.0)
    keywords: List[str] = Field(
        default_factory=list, description="Top keywords in cluster"
    )
    color: str = Field(default="#58a6ff", description="Display color for this cluster")


class GapInfo(BaseModel):
    """
    Information about a detected gap (sparse region).
    """

    model_config = ConfigDict(extra="forbid")

    gap_id: int
    description: str = Field(default="", description="What's missing")
    x: float = Field(default=0.0, description="Center of gap region")
    y: float = Field(default=0.0)
    severity: str = Field(default="medium", description="low, medium, or high")
    isolated_chunk_ids: List[str] = Field(default_factory=list)


class MapStats(BaseModel):
    """
    Overall statistics for the knowledge map.
    """

    model_config = ConfigDict(extra="forbid")

    total_chunks: int = Field(default=0)
    total_documents: int = Field(default=0)
    num_clusters: int = Field(default=0)
    num_gaps: int = Field(default=0)
    avg_cluster_size: float = Field(default=0.0)
    coverage_score: float = Field(
        default=1.0, description="1.0 = no gaps, 0.0 = all gaps"
    )


class KnowledgeMapState(BaseModel):
    """
    Root state model for .fitz/knowledge_map.json.

    Caches embeddings (not projections) for incremental updates.
    Projections, clusters, and gaps are recomputed fresh each run.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1)
    collection: str = Field(..., description="Vector DB collection name")
    embedding_id: str = Field(
        ..., description="Embedding config ID for cache invalidation"
    )
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Cached data (persisted)
    chunks: Dict[str, ChunkEmbedding] = Field(
        default_factory=dict, description="Chunk embeddings keyed by chunk_id"
    )
    documents: Dict[str, DocumentNode] = Field(
        default_factory=dict, description="Document nodes keyed by doc_id"
    )

    # Last projection results (recomputed each run, not essential to cache)
    clusters: List[ClusterInfo] = Field(default_factory=list)
    gaps: List[GapInfo] = Field(default_factory=list)
    stats: MapStats = Field(default_factory=MapStats)

    def get_cached_chunk_ids(self) -> set[str]:
        """Get set of chunk IDs already in cache."""
        return set(self.chunks.keys())

    def add_chunk(self, chunk: ChunkEmbedding) -> None:
        """Add a chunk embedding to the cache."""
        self.chunks[chunk.chunk_id] = chunk

        # Update document node
        if chunk.doc_id not in self.documents:
            self.documents[chunk.doc_id] = DocumentNode(
                doc_id=chunk.doc_id,
                label=chunk.doc_id,
                chunk_ids=[],
            )
        if chunk.chunk_id not in self.documents[chunk.doc_id].chunk_ids:
            self.documents[chunk.doc_id].chunk_ids.append(chunk.chunk_id)

    def remove_stale_chunks(self, current_chunk_ids: set[str]) -> int:
        """
        Remove chunks that no longer exist in the vector DB.

        Returns count of removed chunks.
        """
        stale_ids = set(self.chunks.keys()) - current_chunk_ids
        for chunk_id in stale_ids:
            chunk = self.chunks.pop(chunk_id)
            # Remove from document
            if chunk.doc_id in self.documents:
                doc = self.documents[chunk.doc_id]
                if chunk_id in doc.chunk_ids:
                    doc.chunk_ids.remove(chunk_id)
                # Remove empty documents
                if not doc.chunk_ids:
                    del self.documents[chunk.doc_id]

        return len(stale_ids)


__all__ = [
    "ChunkEmbedding",
    "DocumentNode",
    "ClusterInfo",
    "GapInfo",
    "MapStats",
    "KnowledgeMapState",
]
