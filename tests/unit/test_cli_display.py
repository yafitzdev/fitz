# tests/unit/test_cli_display.py
"""Tests for CLI display helpers."""

from __future__ import annotations

from fitz_sage.cli.ui.display import (
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


def test_evidence_title_combines_pyrrho_verdict():
    """Evidence table title should include the Pyrrho verdict."""
    metadata = {
        "governance_cutoff": {
            "pyrrho": {"mode": "trustworthy"},
        }
    }

    assert _evidence_title("trustworthy", metadata) == "Evidence - Pyrrho trustworthy"


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
        "Pyrrho probabilities: trustworthy=0.53  abstain=0.21  disputed=0.26",
        (
            "Governance cutoff: selected=6  evaluated=6/10  shape=broad  "
            "min_trust=4  min_dispute=2  dispute_patience=2"
        ),
        "Pyrrho: sources support a confident answer (P=0.53).",
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
