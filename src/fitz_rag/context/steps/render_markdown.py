from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .normalize import _to_chunk_dict


@dataclass
class RenderMarkdownStep:
    """
    Render a list of canonical chunks into a markdown string.

    Format per chunk:

        ### Source: <file>
        <text>

    The 'file' is resolved from metadata["file"] if present,
    otherwise falls back to "unknown".
    """

    def __call__(self, chunks: List[Any]) -> str:
        final_sections: List[str] = []

        for ch in chunks:
            c = _to_chunk_dict(ch)
            meta = c.get("metadata", {}) or {}
            file_val = meta.get("file") or "unknown"
            text = c.get("text", "")

            section = (
                f"### Source: {file_val}\n"
                f"{text}\n"
            )
            final_sections.append(section)

        return "\n".join(final_sections)
