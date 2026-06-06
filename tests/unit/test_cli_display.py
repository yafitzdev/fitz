# tests/unit/test_cli_display.py
"""Tests for CLI display helpers."""

from __future__ import annotations

from fitz_sage.cli.ui.display import (
    _compact_evidence_excerpt,
    _evidence_title,
    _format_governance_metadata,
    _format_indexing_status,
)


def test_format_indexing_status_shows_deep_enrichment_when_query_ready():
    """Query-ready collections should not look like indexing is still blocked."""
    status = {
        "total": 63,
        "pending": 0,
        "complete": True,
        "query_ready": True,
        "deep_pending": 62,
        "fully_enriched": False,
    }

    assert _format_indexing_status(status) == "Deep enrichment pending: 62/63"


def test_format_indexing_status_names_single_deep_pending_file():
    """A one-file deep-enrichment tail should be visible to the user."""
    status = {
        "total": 62,
        "pending": 0,
        "complete": True,
        "query_ready": True,
        "deep_pending": 1,
        "deep_pending_files": [
            {"path": "pdf/1.0 BA Yan Fitzner.pdf", "state": "query_ready", "priority": 4}
        ],
        "fully_enriched": False,
    }

    assert _format_indexing_status(status) == (
        "Deep enrichment pending: 1/62 (pdf/1.0 BA Yan Fitzner.pdf, query_ready)"
    )


def test_format_indexing_status_shows_enrichment_after_parse_surface_ready():
    """Parsed files are searchable, so remaining keyword work is enrichment."""
    status = {
        "total": 63,
        "indexed": 1,
        "pending": 62,
        "complete": False,
        "query_ready": False,
        "deep_pending": 63,
        "fully_enriched": False,
        "by_state": {"parsed": 62, "query_ready": 1},
    }

    assert _format_indexing_status(status) == "Enrichment pending: 62/63"


def test_format_indexing_status_shows_indexing_when_query_surface_is_pending():
    """Before query-ready indexing completes, keep the indexing language."""
    status = {
        "total": 63,
        "pending": 12,
        "complete": False,
        "query_ready": False,
        "deep_pending": 63,
        "fully_enriched": False,
    }

    assert _format_indexing_status(status) == "Indexing pending: 12/63"


def test_evidence_title_stays_stable_for_pyrrho_verdict():
    """Evidence table title should not encode governance state."""
    metadata = {
        "governance_cutoff": {
            "pyrrho": {"mode": "trustworthy"},
        }
    }

    assert _evidence_title("trustworthy", metadata) == "Evidence"


def test_broad_overview_title_stays_stable():
    """Broad overview semantics belong in governance metadata, not the title."""
    metadata = {
        "governance_cutoff": {
            "representative_sources": True,
            "policy": {"query_shape": "broad_overview"},
        }
    }

    assert _evidence_title("abstain", metadata) == "Evidence"


def test_format_governance_metadata_shows_pyrrho_and_cutoff():
    """Governance metadata should expose probabilities and cutoff policy."""
    metadata = {
        "governance_cutoff": {
            "evaluated": 6,
            "selected": 6,
            "max": 10,
            "mode": "trustworthy",
            "policy": {
                "query_shape": "broad",
                "min_trustworthy_docs": 4,
                "min_disputed_docs": 2,
                "disputed_patience_docs": 2,
            },
            "pyrrho": {
                "mode": "trustworthy",
                "probabilities": {
                    "abstain": 0.21,
                    "disputed": 0.26,
                    "trustworthy": 0.53,
                },
                "reason": "Pyrrho: sources support a confident answer (P=0.53).",
            },
        }
    }

    assert _format_governance_metadata(metadata, []) == [
        "Pyrrho: TRUSTWORTHY  P(TRUSTWORTHY)=0.53  P(ABSTAIN)=0.21  P(DISPUTED)=0.26",
        (
            "Cutoff: selected 6; evaluated 6/10; policy broad; "
            "min trustworthy 4; min disputed 2; dispute patience 2"
        ),
        "Pyrrho: sources support a confident answer (P=0.53).",
    ]


def test_format_governance_metadata_shows_pyrrho_heads_and_scalars():
    """Pyrrho metadata should be visible without dumping raw JSON."""
    metadata = {
        "governance_cutoff": {
            "pyrrho": {
                "query_contract": {
                    "final_label": "structured_lookup",
                    "confidence": 0.88,
                },
                "route": {"final_label": "business_ops", "confidence": 0.81},
                "taxonomy": {"final_label": "direct_evidence", "confidence": 0.77},
                "retrieval_action": {"final_label": "answer_now", "confidence": 0.83},
                "gap_type": {"final_label": "none", "confidence": 0.86},
                "answerability_shape": {"final_label": "direct_answer", "confidence": 0.76},
                "retrieval_modality": {"final_label": "unstructured_text", "confidence": 0.70},
                "scalars": {
                    "evidence_sufficiency": 0.84,
                    "query_evidence_alignment": 0.79,
                    "retrieval_retry_value": 0.17,
                    "false_trustworthy_risk": 0.09,
                    "evidence_failure_severity": 0.11,
                },
            },
        }
    }

    assert _format_governance_metadata(metadata, []) == [
        (
            "Pyrrho heads: contract structured_lookup (0.88); "
            "route business_ops (0.81); taxonomy direct_evidence (0.77); "
            "action answer_now (0.83); gap none (0.86); "
            "shape direct_answer (0.76); modality unstructured_text (0.70)"
        ),
        (
            "Pyrrho signals: sufficiency 0.84; alignment 0.79; retry 0.17; "
            "false-trust risk 0.09; failure severity 0.11"
        ),
    ]


def test_format_governance_metadata_preserves_extra_reasons():
    """Additional governance reasons should not be hidden by Pyrrho metadata."""
    metadata = {
        "governance_cutoff": {
            "pyrrho": {
                "mode": "abstain",
                "reason": "Pyrrho: retrieved sources do not contain enough evidence (P=0.70).",
            },
        }
    }

    assert _format_governance_metadata(
        metadata,
        [
            "Pyrrho: retrieved sources do not contain enough evidence (P=0.70).",
            "Pyrrho abstained after evaluating the top 10 evidence item(s).",
        ],
    ) == [
        "Pyrrho: retrieved sources do not contain enough evidence (P=0.70).",
        "Pyrrho abstained after evaluating the top 10 evidence item(s).",
    ]


def test_format_governance_metadata_shows_structured_lookup_contract_first():
    """Exact source-finding success should be visible before Pyrrho probabilities."""
    metadata = {
        "governance_cutoff": {
            "structured_lookup_contract": {
                "matched_identifiers": ["TC-0901"],
                "matched_sources": 1,
            },
            "pyrrho": {
                "mode": "abstain",
                "probabilities": {
                    "abstain": 0.73,
                    "disputed": 0.11,
                    "trustworthy": 0.16,
                },
            },
        }
    }

    lines = _format_governance_metadata(
        metadata,
        ["Structured lookup contract satisfied by exact identifier match: TC-0901."],
    )

    assert lines[0] == (
        "Structured lookup: retrieval contract satisfied by exact match for "
        "TC-0901; selected 1 source(s)."
    )
    assert lines[1].startswith("Pyrrho: ABSTAIN")


def test_format_broad_overview_metadata_skips_pyrrho_cutoff_language():
    """Broad overview metadata should explain representative-source semantics."""
    metadata = {
        "governance_cutoff": {
            "evaluated": 0,
            "selected": 4,
            "max": 10,
            "mode": "abstain",
            "representative_sources": True,
            "sufficiency_evaluated": False,
            "policy": {
                "query_shape": "broad_overview",
                "min_trustworthy_docs": 4,
                "min_disputed_docs": 2,
                "disputed_patience_docs": 2,
            },
        }
    }

    assert _format_governance_metadata(
        metadata,
        ["Query is too broad for evidence sufficiency; returned representative sources."],
    ) == [
        (
            "Broad overview: selected 4 representative source(s) from top 10; "
            "evidence sufficiency was not evaluated."
        ),
        "Query is too broad for evidence sufficiency; returned representative sources.",
    ]


def test_compact_evidence_excerpt_keeps_terminal_table_short():
    """Display excerpts should stay compact without changing stored evidence content."""
    text = " ".join(["cell"] * 80)

    excerpt = _compact_evidence_excerpt(text, max_chars=60)

    assert len(excerpt) <= 60
    assert excerpt.endswith("...")
