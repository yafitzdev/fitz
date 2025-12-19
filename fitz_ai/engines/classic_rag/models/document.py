# pipeline/models/document.py
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class Document(BaseModel):
    """
    Canonical document model used across the fitz stack.
    """

    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Full document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
