# benchmarks/corpora/holdout/code/rollout_guard.py
"""Deployment guard used by the holdout benchmark."""

from __future__ import annotations

REQUIRED_DEPLOY_ENV = ("FITZ_DEPLOY_APPROVER", "FITZ_CHANGE_TICKET")


def should_auto_rollback(error_rate_percent: float, severity: str) -> bool:
    """Return whether a deployment should automatically roll back."""
    return error_rate_percent >= 2.5 or severity in {"P0", "P1"}


def canary_window_minutes(region: str) -> int:
    """Return the canary observation window for a region."""
    if region == "apac":
        return 45
    return 30
