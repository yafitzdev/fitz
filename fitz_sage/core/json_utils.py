# fitz_sage/core/json_utils.py
"""Parse JSON values out of LLM responses.

LLMs wrap JSON in ```json fences, prepend prose, or emit it bare. ``parse_llm_json``
is the single place that pulls a dict (or list) back out — used by query batching,
detection, enrichment, and the retrieval strategies.
"""

from __future__ import annotations

import json
from typing import Any


def parse_llm_json(response: str, *, as_array: bool = False) -> Any:
    """Extract a JSON object (or array) from an LLM response.

    Tries, in order: a ```json fenced block, the whole response, then a JSON
    value embedded in surrounding prose.

    Args:
        response: Raw LLM output.
        as_array: Parse a JSON array; otherwise a JSON object.

    Returns:
        The parsed ``list`` (when ``as_array``) or ``dict`` — or an empty one
        of that type when nothing parseable is found.
    """
    text = response.strip()
    want: type = list if as_array else dict
    open_ch = "[" if as_array else "{"

    # 1. Fenced ```json block.
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith(open_ch):
                try:
                    parsed = json.loads(part)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, want):
                    return parsed

    # 2. The whole response is JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, want):
            return parsed
    except json.JSONDecodeError:
        pass

    # 3. A JSON value embedded in surrounding prose.
    if as_array:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
    else:
        start = text.find("{")
        if start >= 0:
            depth = 0
            for i, char in enumerate(text[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start : i + 1])
                            if isinstance(parsed, dict):
                                return parsed
                        except json.JSONDecodeError:
                            pass
                        break

    return [] if as_array else {}
