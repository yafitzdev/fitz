# tools/retrieval_eval/matchers.py
"""Match retrieved ReadResults against ground-truth units, per retrieval mode.

For one query a matcher answers: for each ground-truth unit, what is the
1-indexed rank of the first retrieved result that satisfies it? That rank
(or None) plus the unit's grade is all metrics.py needs.

Matching is mode-specific because the three modes retrieve different things:
  code    -> a file path
  section -> a heading or page within a document
  table   -> a cell value from a table
"""

from __future__ import annotations

from typing import Any

from .metrics import Unit


def _norm(text: Any) -> str:
    """Lowercase and strip all whitespace — tolerant heading comparison."""
    return "".join(str(text).lower().split())


def _flatten(read_result: Any) -> dict:
    """Collapse a ReadResult and its Address into one lookup dict."""
    fields = dict(getattr(read_result, "metadata", None) or {})
    address = getattr(read_result, "address", None)
    if address is not None:
        fields.update(getattr(address, "metadata", None) or {})
        fields["location"] = getattr(address, "location", "") or ""
    fields["file_path"] = getattr(read_result, "file_path", "") or ""
    fields["content"] = getattr(read_result, "content", "") or ""
    return fields


def _match_code(fields: dict, unit: dict) -> bool:
    """A code unit matches when the result's file is the unit's file.

    Suffix match in both directions, so it holds whether the engine reports
    absolute, repo-relative, or corpus-relative file paths.
    """
    want = unit["path"].replace("\\", "/").lower()
    got = str(fields["file_path"]).replace("\\", "/").lower()
    if not want or not got:
        return False
    return got.endswith(want) or want.endswith(got)


def _match_section(fields: dict, unit: dict) -> bool:
    """A section unit matches on heading text or page proximity, same document."""
    doc = unit["doc"].lower()
    location = str(fields.get("location", "")).lower()
    file_path = str(fields["file_path"]).lower()
    if doc not in file_path and doc not in location:
        return False

    heading = _norm(unit.get("heading", ""))
    if heading:
        for key in ("heading", "section_title", "name", "location"):
            if heading in _norm(fields.get(key, "")):
                return True

    page = unit.get("page")
    if page is not None:
        for key in ("page_start", "page", "page_end"):
            value = fields.get(key)
            if value is not None and abs(int(value) - int(page)) <= 1:
                return True
    return False


def _match_table(fields: dict, unit: dict) -> bool:
    """A table unit matches when the cell value appears in a result from the table."""
    doc = unit["doc"].lower()
    file_path = str(fields["file_path"]).lower()
    location = str(fields.get("location", "")).lower()
    if doc not in file_path and doc not in location:
        return False
    return str(unit["value"]).lower() in str(fields["content"]).lower()


_MATCHERS = {
    "code": _match_code,
    "section": _match_section,
    "table": _match_table,
    "query_profile": _match_code,
}


def rank_units(mode: str, read_results: list, units: list[dict]) -> list[Unit]:
    """Rank each ground-truth unit against one query's retrieved results.

    Args:
        mode: one of "code", "section", "table".
        read_results: ranked ReadResults from ``engine.retrieve()``.
        units: ground-truth unit dicts for the query.

    Returns:
        One Unit(grade, rank) per ground-truth unit, in input order.
    """
    match = _MATCHERS[mode]
    flattened = [_flatten(r) for r in read_results]
    ranked: list[Unit] = []
    for unit in units:
        rank = None
        for position, fields in enumerate(flattened, start=1):
            if match(fields, unit):
                rank = position
                break
        ranked.append(Unit(grade=unit["grade"], rank=rank))
    return ranked
