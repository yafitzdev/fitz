# fitz_sage/retrieval/rewriter/rewriter.py
"""Query-rewrite result parsing.

Query rewriting runs as a section of the batched query-prep call
(``QueryBatcher``). This module holds only the shared parser that turns
the rewriting section's JSON into a ``RewriteResult``.
"""

from __future__ import annotations

from .types import RewriteResult, RewriteType


def parse_rewrite_dict(data: dict, original_query: str) -> RewriteResult:
    """Parse a rewriting-section dict (already JSON-decoded) into a RewriteResult."""
    rewritten = data.get("rewritten_query", original_query)
    rewrite_type_str = data.get("rewrite_type", "none")
    confidence = float(data.get("confidence", 0.5))
    is_ambiguous = data.get("is_ambiguous", False)
    disambiguated = data.get("disambiguated_queries", [])
    is_compound = data.get("is_compound", False)
    decomposed = data.get("decomposed_queries", [])

    type_mapping = {
        "none": RewriteType.NONE,
        "conversational": RewriteType.CONVERSATIONAL,
        "clarity": RewriteType.CLARITY,
        "retrieval": RewriteType.RETRIEVAL,
        "decomposition": RewriteType.DECOMPOSITION,
        "combined": RewriteType.COMBINED,
    }
    rewrite_type = type_mapping.get(str(rewrite_type_str).lower(), RewriteType.NONE)

    if not rewritten or not str(rewritten).strip():
        rewritten = original_query
        rewrite_type = RewriteType.NONE

    return RewriteResult(
        original_query=original_query,
        rewritten_query=str(rewritten).strip(),
        rewrite_type=rewrite_type,
        confidence=confidence,
        is_ambiguous=is_ambiguous,
        disambiguated_queries=disambiguated[:3],
        is_compound=is_compound,
        decomposed_queries=decomposed[:5],
    )
