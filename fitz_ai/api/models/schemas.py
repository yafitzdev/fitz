# fitz_ai/api/models/schemas.py
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
    collection: str = Field("default", description="Collection to query")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve", ge=1)


class QueryResponse(BaseModel):
    """Response from a knowledge base query."""

    text: str = Field(..., description="The answer text")
    mode: Optional[str] = Field(
        None, description="Answer mode: confident, qualified, disputed, or abstain"
    )
    sources: List[SourceInfo] = Field(
        default_factory=list, description="Sources used in the answer"
    )


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
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve", ge=1)


class ChatResponse(BaseModel):
    """Response from a chat request."""

    text: str = Field(..., description="The assistant's response")
    mode: Optional[str] = Field(
        None, description="Answer mode: confident, qualified, disputed, or abstain"
    )
    sources: List[SourceInfo] = Field(
        default_factory=list, description="Sources used in the response"
    )


class IngestRequest(BaseModel):
    """Request to ingest documents."""

    source: str = Field(..., description="Path to file or directory to ingest")
    collection: str = Field("default", description="Target collection name")
    clear_existing: bool = Field(False, description="Clear existing collection before ingesting")


class IngestResponse(BaseModel):
    """Response from an ingestion request."""

    documents: int = Field(..., description="Number of documents ingested")
    chunks: int = Field(..., description="Number of chunks created")
    collection: str = Field(..., description="Target collection name")


class CollectionInfo(BaseModel):
    """Basic information about a collection."""

    name: str = Field(..., description="Collection name")
    chunk_count: int = Field(..., description="Number of chunks in the collection")


class CollectionStats(BaseModel):
    """Detailed statistics for a collection."""

    name: str = Field(..., description="Collection name")
    chunk_count: int = Field(..., description="Number of chunks")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional collection metadata"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status: healthy or unhealthy")
    version: str = Field(..., description="Fitz version")
    config_exists: bool = Field(..., description="Whether config file exists")
