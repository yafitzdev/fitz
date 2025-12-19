# pipeline/generation/prompting/profiles.py
from __future__ import annotations

from enum import Enum


class PromptProfile(str, Enum):
    """
    Prompt profile identifier.

    A profile defines prompt authority boundaries.
    At the moment, only RAG_USER exists and is a no-op profile
    matching current behavior exactly.
    """

    RAG_USER = "rag_user"
