# src/fitz_rag/pipeline/rag_pipeline.py
"""
High-level RAG pipeline for fitz-rag.

This module glues together:
- RAGContextBuilder (multi-source retrieval)
- Prompt builder (TRF + RAG context + task)
- Chat client (LLM provider)

It exposes a small function to perform a single analysis run.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from fitz_rag.sourcer.rag_base import (
    RAGContextBuilder,
    SourceConfig,
)
from fitz_rag.sourcer.prompt_builder import build_user_prompt
from fitz_rag.llm.chat_client import ChatClient

# NEW – unified config to provide a fallback for max_trf_json_chars
from fitz_rag.config import get_config
_cfg = get_config()


def run_single_rag_analysis(
    trf: Dict,
    query: str,
    task_prompt: str,
    system_prompt: str,
    sources: List[SourceConfig],
    context_builder: RAGContextBuilder,
    chat_client: ChatClient,
    max_trf_json_chars: Optional[int] = None,
) -> str:
    """
    Perform a single RAG analysis run:

    - Use `query` for retrieval
    - Use `trf` as the primary JSON context
    - Use `sources` and `context_builder` to build a RetrievalContext
    - Use `task_prompt` as the task description (what the model should do)
    - Use `system_prompt` as the LLM system role definition

    Returns the model's answer as a string.
    """

    # apply config fallback
    if max_trf_json_chars is None:
        max_trf_json_chars = _cfg.get("retriever", {}).get("max_trf_json_chars", None)

    # 1) Run multi-source retrieval
    ctx = context_builder.retrieve_for(trf=trf, query=query)

    # 2) Build user message content
    user_content = build_user_prompt(
        trf=trf,
        ctx=ctx,
        prompt_text=task_prompt,
        sources=sources,
        max_trf_json_chars=max_trf_json_chars,
    )

    # 3) Call the LLM
    answer = chat_client.chat(
        system_prompt=system_prompt,
        user_content=user_content,
    )

    return answer
