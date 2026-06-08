# benchmarks/corpora/holdout2/code/pricing_engine.py
"""Pricing helpers used by the second holdout benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass

DISCOUNT_OVERRIDE_ENV = "FITZ_HOLDOUT2_DISCOUNT_OVERRIDE"


@dataclass(frozen=True)
class InvoiceCharge:
    """Invoice charge calculated for renewal."""

    amount_usd: float
    discount_percent: float


def enterprise_discount(account_tier: str, seats: int) -> float:
    """Return the enterprise renewal discount percent."""
    if os.environ.get(DISCOUNT_OVERRIDE_ENV):
        return 0.25
    if account_tier == "platinum" and seats >= 1000:
        return 0.18
    if account_tier == "gold" and seats >= 500:
        return 0.10
    return 0.0


def renewal_notice_days(vendor_category: str) -> int:
    """Return the renewal notice period by vendor category."""
    if vendor_category == "edge":
        return 90
    if vendor_category == "model_eval":
        return 75
    return 30
