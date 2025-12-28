# fitz_ai/prompts/hierarchy/__init__.py
"""
Hierarchy prompts for multi-level summarization.

Prompts:
- GROUP_SUMMARY_PROMPT: Summarizes chunks within a group (e.g., per-file)
- CORPUS_SUMMARY_PROMPT: Synthesizes insights across all groups
- build_epistemic_group_prompt: Epistemic-aware group summarization
- build_epistemic_corpus_prompt: Epistemic-aware corpus summarization
"""

from fitz_ai.prompts.hierarchy.corpus_summary import PROMPT as CORPUS_SUMMARY_PROMPT
from fitz_ai.prompts.hierarchy.corpus_summary_epistemic import (
    build_epistemic_corpus_context,
    build_epistemic_corpus_prompt,
)
from fitz_ai.prompts.hierarchy.group_summary import PROMPT as GROUP_SUMMARY_PROMPT
from fitz_ai.prompts.hierarchy.group_summary_epistemic import build_epistemic_group_prompt

__all__ = [
    "GROUP_SUMMARY_PROMPT",
    "CORPUS_SUMMARY_PROMPT",
    "build_epistemic_group_prompt",
    "build_epistemic_corpus_prompt",
    "build_epistemic_corpus_context",
]
