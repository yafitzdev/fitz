# fitz_sage/api/models/schemas.py
"""Pydantic models for API requests and responses."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """Information about a source used in an answer."""

    source_id: str = Field(..., description="Unique identifier for the source")
    excerpt: Optional[str] = Field(None, description="Relevant excerpt from the source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional source metadata")


class QueryRequest(BaseModel):
    """Request to query the knowledge base."""

    question: str = Field(..., description="The question to ask", min_length=1)
    source: Optional[str] = Field(
        None,
        description="Path to file or directory. If provided, registers documents before querying.",
    )
    collection: str = Field("default", description="Collection to query")
    conversation_history: List["ChatMessage"] = Field(
        default_factory=list,
        description="Optional conversation history for query rewriting (resolves pronouns like 'their' → 'TechCorp')",
    )


class QueryResponse(BaseModel):
    """Response from a knowledge base query."""

    text: str = Field(..., description="The answer text")
    mode: Optional[str] = Field(
        None, description="Answer mode: sufficient, disputed, or insufficient"
    )
    sources: List[SourceInfo] = Field(
        default_factory=list, description="Sources used in the answer"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra answer metadata, e.g. gap_context (what's missing / what to add) on insufficient",
    )


class EvidenceItemResponse(BaseModel):
    """One ranked evidence item returned by retrieval-first endpoints."""

    rank: int = Field(..., description="Rank within the selected evidence prefix")
    source_id: str = Field(..., description="Stable source identifier")
    file_path: str = Field(..., description="Source file path")
    address_kind: str = Field(..., description="Typed unit kind: section, symbol, table, or file")
    address_location: str = Field(..., description="Location within the source")
    line_range: Optional[List[int]] = Field(None, description="Line range when available")
    score: Optional[float] = Field(None, description="Retrieval or rerank score")
    excerpt: str = Field(..., description="Display excerpt")
    content: str = Field(..., description="Full evidence content passed to Pyrrho")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Item metadata")


class EvidenceResponse(BaseModel):
    """Retrieval-first response from the evidence endpoint."""

    query: str = Field(..., description="Original query text")
    mode: Optional[str] = Field(None, description="Governance mode")
    items: List[EvidenceItemResponse] = Field(default_factory=list, description="Evidence items")
    reasons: List[str] = Field(default_factory=list, description="Governance reasons")
    timings: Dict[str, Any] = Field(default_factory=dict, description="Stage timings")
    indexing_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Progressive indexing status for the collection",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Governance metadata")


class ChatMessage(BaseModel):
    """A message in a chat conversation."""

    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for multi-turn chat."""

    message: str = Field(..., description="The current user message", min_length=1)
    history: List[ChatMessage] = Field(
        default_factory=list, description="Previous conversation messages"
    )
    collection: str = Field("default", description="Collection to query")


class ChatResponse(BaseModel):
    """Response from a chat request."""

    text: str = Field(..., description="The assistant's response")
    mode: Optional[str] = Field(
        None, description="Answer mode: sufficient, disputed, or insufficient"
    )
    sources: List[SourceInfo] = Field(
        default_factory=list, description="Sources used in the response"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra answer metadata, e.g. gap_context (what's missing / what to add) on INSUFFICIENT",
    )


class IngestRequest(BaseModel):
    """Request to register documents into a collection."""

    source: str = Field(..., description="Path to a file or directory to ingest", min_length=1)


class IndexingStatus(BaseModel):
    """Background-indexing progress for a collection."""

    discovered: int = Field(0, description="All files discovered under the source")
    total: int = Field(..., description="Supported files, including failed files")
    indexed: int = Field(..., description="Supported files that are query-ready")
    pending: int = Field(..., description="Supported files still being indexed")
    failed: int = Field(0, description="Supported files that failed indexing")
    failed_files: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-file indexing failures with stage and error",
    )
    unsupported: int = Field(0, description="Files outside the enabled format contract")
    unsupported_files: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Unsupported file paths and extensions",
    )
    healthy: bool = Field(True, description="True when no supported file failed")
    complete: bool = Field(
        ...,
        description="True when every supported file indexed successfully",
    )
    query_ready: bool = Field(
        False,
        description="True when no supported file remains pending",
    )
    deep_pending: int = Field(0, description="Files awaiting deep enrichment")
    deep_pending_files: List[Dict[str, Any]] = Field(default_factory=list)
    fully_enriched: bool = Field(False)
    by_state: Dict[str, int] = Field(
        default_factory=dict, description="File counts per indexing state"
    )


class CollectionInfo(BaseModel):
    """Basic information about a collection."""

    name: str = Field(..., description="Collection name")
    item_count: int = Field(..., description="Number of items in the collection")


class CollectionStats(BaseModel):
    """Detailed statistics for a collection."""

    name: str = Field(..., description="Collection name")
    item_count: int = Field(..., description="Number of items")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional collection metadata"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status: healthy or unhealthy")
    version: str = Field(..., description="Fitz version")
    components: Dict[str, bool] = Field(default_factory=dict, description="Component health status")
