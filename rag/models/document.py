from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Optional


class Document(BaseModel):
    """
    Base document model stored before or during chunking.

    Validation errors here propagate as Pydantic exceptions,
    which upper layers convert into ConfigError.
    """
    id: str = Field(..., description="Unique identifier for the document.")
    path: Optional[str] = Field(None, description="Optional file path.")
    metadata: Dict = Field(default_factory=dict, description="Document-level metadata.")
    content: str = Field(..., description="Full raw text of the document.")
