# src/fitz_rag/sourcer/prompt_builder.py
"""
Prompt builder utilities for fitz-rag.

This module converts:
    - TRF JSON
    - RetrievalContext
    - List[SourceConfig]
    - Prompt text

into a unified user message to send to the LLM.
"""

from __future__ import annotations

import json
from typing import Dict, List

from fitz_rag.core import RetrievedChunk
from fitz_rag.sourcer.rag_base import RetrievalContext, SourceConfig

# NEW — unified config fallback for JSON truncation
from fitz_rag.config import get_config
_cfg = get_config()


def _format_chunks(label: str, chunks: List[RetrievedChunk]) -> str:
    if not chunks:
        return f"{label}: <no chunks>"

    out = [f"{label} (top {len(chunks)}):"]
    for idx, c in enumerate(chunks, 1):
        src = c.metadata.get("file") or c.metadata.get("source") or ""
        out.append(f"[{idx}] score={c.score:.4f} | {src}")
        out.append(c.text.strip())
        out.append("")

    return "\n".join(out).strip()


def build_rag_block(ctx: RetrievalContext, sources: List[SourceConfig]) -> str:
    parts = [f"Retrieval query: {ctx.query}", ""]

    # Known artefacts in declared order
    for src in sources:
        label = src.label or src.name.upper()
        chunks = ctx.artefacts.get(src.name, [])
        parts.append(_format_chunks(label, chunks))
        parts.append("")

    return "\n".join(parts).strip()


def build_user_prompt(
    trf: Dict,
    ctx: RetrievalContext,
    prompt_text: str,
    sources: List[SourceConfig],
    max_trf_json_chars: int | None = None,
) -> str:

    # CONFIG FALLBACK — if not supplied, pull from YAML
    if max_trf_json_chars is None:
        max_trf_json_chars = _cfg.get("retriever", {}).get("max_trf_json_chars", None)

    trf_json = json.dumps(trf, indent=2)

    if max_trf_json_chars and len(trf_json) > max_trf_json_chars:
        trf_json = trf_json[:max_trf_json_chars] + "\n...[TRF JSON truncated]..."

    rag_block = build_rag_block(ctx, sources)

    return (
        f"=== TRF JSON ===\n{trf_json}\n\n"
        f"=== RAG CONTEXT ===\n{rag_block}\n\n"
        f"=== TASK ===\n{prompt_text}\n"
    )
