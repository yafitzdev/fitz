# benchmarks/corpora/holdout/code/alert_router.py
"""Alert routing rules used by the holdout benchmark."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_SLACK_CHANNEL = "#ops-alerts"


@dataclass(frozen=True)
class AlertRoute:
    """Resolved routing target for an alert."""

    target: str
    notify_owner: bool


def route_for_severity(severity: str) -> AlertRoute:
    """Resolve the notification route for an alert severity."""
    if severity in {"P0", "P1"}:
        return AlertRoute(target="pager", notify_owner=True)
    if severity == "P2":
        return AlertRoute(target=DEFAULT_SLACK_CHANNEL, notify_owner=True)
    return AlertRoute(target="ticket", notify_owner=False)


def should_page_owner(alert: dict[str, str]) -> bool:
    """Return whether the owner should be paged for an alert."""
    return route_for_severity(alert["severity"]).target == "pager"
