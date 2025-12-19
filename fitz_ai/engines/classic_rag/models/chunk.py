# pipeline/models/chunk.py
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Canonical chunk model used across the entire fitz stack.
    """

    id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Index of this chunk within its document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
