"""Input and output contract of the managed Pyrrho ONNX model."""

from __future__ import annotations

from collections.abc import Iterable

EVIDENCE_VERDICTS: tuple[str, ...] = ("INSUFFICIENT", "DISPUTED", "SUFFICIENT")
FAILURE_MODES: tuple[str, ...] = (
    "none",
    "unresolved_conflict",
    "missing_or_incomplete_evidence",
    "wrong_scope_or_version",
    "ambiguous_request",
)
RETRIEVAL_INTENT_KEYS: tuple[str, ...] = (
    "needs_lookup",
    "needs_temporal_resolution",
    "needs_comparison_or_set",
    "needs_broad_coverage",
)
EVIDENCE_KIND_KEYS: tuple[str, ...] = (
    "needs_text",
    "needs_table_or_record",
    "needs_code_or_symbol",
    "needs_config_or_setting",
    "needs_log_or_run_result",
    "needs_document_layout",
)
NUM_PYRRHO_LABELS = (
    len(EVIDENCE_VERDICTS)
    + len(FAILURE_MODES)
    + len(RETRIEVAL_INTENT_KEYS)
    + len(EVIDENCE_KIND_KEYS)
)

PYRRHO_PRE_TAG = "[PYRRHO_PRE]"
PYRRHO_POST_TAG = "[PYRRHO_POST]"


def build_pyrrho_evidence_text(
    query: str,
    contexts: Iterable[dict[str, str] | str],
) -> str:
    """Build the model's post-retrieval input without changing evidence text."""
    parts = [PYRRHO_POST_TAG, f"Question: {(query or '').strip()}", "", "Sources:"]
    for index, context in enumerate(contexts or [], start=1):
        if isinstance(context, dict):
            source_id = str(context.get("source_id") or index)
            text = str(context.get("text") or "").strip()
        else:
            source_id = str(index)
            text = str(context).strip()
        parts.append(f"[{source_id}] {text}")
    return "\n".join(parts)


def build_pyrrho_query_text(query: str) -> str:
    """Build the model's pre-retrieval input from only the user query."""
    return f"{PYRRHO_PRE_TAG}\nQuestion: {(query or '').strip()}"


def pyrrho_label_names() -> list[str]:
    """Return the exact ordered names of the model's 18 logits."""
    return (
        [f"evidence_verdict.{name}" for name in EVIDENCE_VERDICTS]
        + [f"failure_mode.{name}" for name in FAILURE_MODES]
        + [f"retrieval_intents.{name}" for name in RETRIEVAL_INTENT_KEYS]
        + [f"evidence_kinds.{name}" for name in EVIDENCE_KIND_KEYS]
    )


__all__ = [
    "EVIDENCE_KIND_KEYS",
    "EVIDENCE_VERDICTS",
    "FAILURE_MODES",
    "NUM_PYRRHO_LABELS",
    "RETRIEVAL_INTENT_KEYS",
    "build_pyrrho_evidence_text",
    "build_pyrrho_query_text",
    "pyrrho_label_names",
]
