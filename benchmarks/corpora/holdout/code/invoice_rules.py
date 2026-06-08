# benchmarks/corpora/holdout/code/invoice_rules.py
"""Invoice rules used by the holdout benchmark."""

from __future__ import annotations

import os

WAIVER_ENV = "FITZ_HOLDOUT_WAIVE_LATE_FEES"


def calculate_late_fee(
    amount_usd: float,
    late_fee_percent: float,
    days_overdue: int,
) -> float:
    """Return the late fee for an overdue invoice."""
    if days_overdue <= 0:
        return 0.0
    if os.environ.get(WAIVER_ENV):
        return 0.0
    return round(amount_usd * late_fee_percent / 100.0, 2)


def collection_owner(status: str, owner: str) -> str:
    """Return the collection owner for an invoice."""
    if status == "overdue":
        return owner
    return "Accounts Payable"
