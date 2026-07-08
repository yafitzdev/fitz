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


def test_format_governance_metadata_shows_query_profile():
    """Pre-retrieval profile knobs should be visible separately from cutoff."""
    metadata = {
        "query_profile": {
            "profile": {
                "specificity": "moderate",
                "answer_type": "comparative",
                "domain": "technical",
                "top_k": 20,
                "top_read": 12,
                "strategy_weights": {"section": 0.25, "code": 0.25, "table": 0.55},
            },
        }
    }

    assert _format_governance_metadata(metadata, []) == [
        (
            "Query profile: profile moderate/comparative/technical; top 20; read 12; "
            "weights section 0.25, code 0.25, table 0.55"
        )
    ]


def test_format_governance_metadata_shows_native_v2_heads():
    """v2 metadata should show native heads only."""
    metadata = {
        "governance_cutoff": {
            "pyrrho": {
                "evidence_verdict": {"final_label": "SUFFICIENT", "confidence": 0.92},
                "failure_mode": {"final_label": "none", "confidence": 0.88},
                "retrieval_intents": {
                    "final_label": "needs_lookup",
                    "final_labels": ["needs_lookup", "needs_temporal_resolution"],
                    "confidence": 0.96,
                },
                "evidence_kinds": {
                    "final_label": "needs_text",
                    "final_labels": ["needs_text"],
                    "confidence": 0.81,
                },
            },
        }
    }

    assert _format_governance_metadata(metadata, []) == [
        (
            "Pyrrho heads: verdict SUFFICIENT (0.92); failure none (0.88); "
            "intents needs_lookup, needs_temporal_resolution (0.96); "
            "evidence needs_text (0.81)"
        )
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
