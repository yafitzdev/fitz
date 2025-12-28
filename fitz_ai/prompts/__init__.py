# fitz_ai/prompts/__init__.py
"""
Centralized prompt library for fitz-ai.

All LLM prompts are defined here for maintainability and versioning.
Prompts are organized by feature area, one file per prompt.

Usage:
    from fitz_ai.prompts import hierarchy

    prompt = hierarchy.GROUP_SUMMARY_PROMPT
"""

from fitz_ai.prompts import hierarchy

__all__ = ["hierarchy"]
