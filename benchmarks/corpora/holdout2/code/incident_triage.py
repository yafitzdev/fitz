# benchmarks/corpora/holdout2/code/incident_triage.py
"""Incident triage helpers used by the second holdout benchmark."""

from __future__ import annotations

ESCALATION_CHANNELS = {
    "S0": "#incident-command",
    "S1": "#payments-war-room",
    "S2": "#ops-alerts",
}


def escalation_channel(severity: str) -> str:
    """Return the escalation channel for an incident severity."""
    return ESCALATION_CHANNELS.get(severity, "#support-triage")


def should_declare_major(duration_minutes: int, affected_customers: int) -> bool:
    """Return whether an incident should be declared major."""
    return duration_minutes >= 30 or affected_customers >= 5000


def customer_visible_label(customer_visible: bool) -> str:
    """Return the public visibility label for an incident."""
    if customer_visible:
        return "customer-visible"
    return "internal-only"
