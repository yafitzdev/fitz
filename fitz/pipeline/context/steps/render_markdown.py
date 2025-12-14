# pipeline/context/steps/render_markdown.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalize import _to_chunk_dict


@dataclass
class RenderMarkdownStep:
    """
    Debug helper: render chunks into a markdown string.

    Not used by the main RAG pipeline.
    """

    def __call__(self, chunks: list[Any]) -> str:
        final_sections: list[str] = []

        for ch in chunks:
            c = _to_chunk_dict(ch)
            doc_id = c.get("doc_id") or "unknown"
            content = c.get("content", "")

            final_sections.append(f"### Source: {doc_id}\n{content}\n")

        return "\n".join(final_sections)
