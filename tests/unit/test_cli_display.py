# tests/unit/test_cli_display.py
"""Tests for CLI display helpers."""

from __future__ import annotations

from fitz_sage.cli.ui.display import _format_indexing_status


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
