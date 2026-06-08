# benchmarks/corpora/holdout2/code/export_scheduler.py
"""Export scheduling helpers used by the second holdout benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass

EXPORT_LOCK_ENV = "FITZ_HOLDOUT2_EXPORT_LOCK"


@dataclass(frozen=True)
class ExportJob:
    """Scheduled export job."""

    export_id: str
    dataset: str
    window_utc: str


def next_export_window_utc(region: str) -> str:
    """Return the next export processing window for a region."""
    if os.environ.get(EXPORT_LOCK_ENV):
        return "locked"
    if region == "eu":
        return "02:00"
    if region == "apac":
        return "16:00"
    return "06:00"


def should_skip_export(row_count: int, encrypted: bool) -> bool:
    """Return whether an export should be skipped for safety."""
    return row_count > 100000 and not encrypted


def redact_destination(destination: str) -> str:
    """Return a redacted storage destination."""
    scheme, _, _rest = destination.partition("://")
    return f"{scheme}://<redacted>"
