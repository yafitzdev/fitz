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
    mode: Optional[str] = Field(None, description="Answer mode: trustworthy, disputed, or abstain")
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


class ChatResponse(BaseModel):
    """Response from a chat request."""

    text: str = Field(..., description="The assistant's response")
    mode: Optional[str] = Field(None, description="Answer mode: trustworthy, disputed, or abstain")
    sources: List[SourceInfo] = Field(
        default_factory=list, description="Sources used in the response"
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
