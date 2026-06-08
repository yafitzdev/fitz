# benchmarks/corpora/holdout2/code/access_control.py
"""Access-control helpers used by the second holdout benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass

ADMIN_OVERRIDE_ENV = "FITZ_HOLDOUT2_ADMIN_OVERRIDE"
SERVICE_TOKEN_ROTATION_DAYS = 45
HUMAN_TOKEN_ROTATION_DAYS = 90


@dataclass(frozen=True)
class ProjectAccess:
    """Export access settings for one project."""

    project_code: str
    allow_exports: bool


class AccessPolicy:
    """Current access policy implementation."""

    def can_export_data(self, user: dict[str, str], project: ProjectAccess) -> bool:
        """Return whether the user may export customer data from a project."""
        if user.get("archived") == "true":
            return False
        if user.get("role") == "admin" and os.environ.get(ADMIN_OVERRIDE_ENV):
            return True
        if user.get("type") == "guest":
            return False
        return project.allow_exports

    def token_rotation_days(self, account_type: str) -> int:
        """Return the token rotation interval for an account type."""
        if account_type == "service":
            return SERVICE_TOKEN_ROTATION_DAYS
        return HUMAN_TOKEN_ROTATION_DAYS


def redact_customer_email(email: str) -> str:
    """Return a redacted customer email address."""
    local, _, domain = email.partition("@")
    return f"{local[:2]}***@{domain}"
