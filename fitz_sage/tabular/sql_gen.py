# fitz_sage/tabular/sql_gen.py
"""Shared SQL-from-LLM helper for the tabular query paths.

Both the query-time ``TableQueryStep`` and the file-path ``DirectTableQuery``
pull a SQL statement out of an LLM response the same way. This is that single
helper. (Prompt templates and result formatting deliberately stay per-class —
they have drifted to serve their distinct query-time vs file contexts.)
"""

from __future__ import annotations

import re


def extract_sql(response: str) -> str:
    """Extract a SQL statement from an LLM response, stripping fences/prose."""
    text = response.strip()

    if "```" in text:
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            text = text.replace("```sql", "").replace("```", "").strip()

    if not text.upper().startswith("SELECT"):
        match = re.search(r"(SELECT\s+.+)", text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1)

    return text
