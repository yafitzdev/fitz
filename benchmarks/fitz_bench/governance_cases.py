"""Balanced fixed-evidence governance benchmark cases.

These cases intentionally bypass live retrieval. They test Pyrrho's core
judgment over a fixed ``(query, evidence)`` input with a 40/40/40 spread across
SUFFICIENT, DISPUTED, and INSUFFICIENT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GovernanceCase:
    """One fixed-evidence Pyrrho benchmark case."""

    case_id: str
    domain: str
    query: str
    expected_mode: str
    contexts: tuple[dict[str, Any], ...]
    tags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "case_id": self.case_id,
            "domain": self.domain,
            "query": self.query,
            "expected_mode": self.expected_mode,
            "contexts": [dict(context) for context in self.contexts],
            "tags": list(self.tags),
        }


_SUFFICIENT = (
    (
        "technology",
        "NimbusDB 4.2.3 ARM median write latency",
        "10.8 ms",
        "benchmark table",
        "table",
    ),
    (
        "technology",
        "OAuthClientV2 retry backoff cap",
        "45 seconds",
        "configuration reference",
        "config",
    ),
    (
        "technology",
        "TC-4812 retry owner",
        "Payments Platform",
        "incident runbook",
        "text",
    ),
    (
        "technology",
        "Atlas feature flag expiry",
        "2026-09-30",
        "feature flag registry",
        "table",
    ),
    (
        "technology",
        "LogStream parser error budget",
        "0.25 percent",
        "SLO table",
        "table",
    ),
    (
        "law_policy",
        "Amber board guest export rule",
        "guest collaborators cannot export customer data",
        "access policy",
        "text",
    ),
    (
        "law_policy",
        "Cobalt CDN breach notice window",
        "30 days",
        "security addendum",
        "text",
    ),
    (
        "law_policy",
        "biometric artifact retention",
        "24 hours",
        "retention schedule",
        "text",
    ),
    (
        "law_policy",
        "MeridianAI renewal notice",
        "75 days written notice",
        "procurement policy",
        "text",
    ),
    (
        "law_policy",
        "service account token rotation",
        "45 days",
        "access control handbook",
        "text",
    ),
    (
        "science_medicine",
        "Trial Zeta cohort B median response",
        "18.4 percent",
        "clinical summary table",
        "table",
    ),
    (
        "science_medicine",
        "Lab assay L-22 storage temperature",
        "-80 C",
        "lab protocol",
        "text",
    ),
    (
        "science_medicine",
        "Specimen batch RX-9 exclusion reason",
        "hemolysis",
        "sample log",
        "log",
    ),
    (
        "science_medicine",
        "Device D17 calibration interval",
        "14 days",
        "calibration SOP",
        "text",
    ),
    (
        "science_medicine",
        "Study Maple enrollment target",
        "420 participants",
        "trial registry",
        "table",
    ),
    (
        "economics_finance",
        "Harborline April standard 30-year rate",
        "6.25 percent",
        "rate sheet",
        "table",
    ),
    (
        "economics_finance",
        "Project Atlas FY27 pipeline",
        "7.8 million ARR",
        "finance forecast",
        "text",
    ),
    (
        "economics_finance",
        "invoice INV-702 amount",
        "27100 USD",
        "invoice table",
        "table",
    ),
    (
        "economics_finance",
        "Northwind CDN annual spend",
        "112000 USD",
        "vendor spend table",
        "table",
    ),
    (
        "economics_finance",
        "Q4 gross margin forecast",
        "63.2 percent",
        "planning workbook",
        "table",
    ),
    (
        "history_geography",
        "Port Selene charter date",
        "1894-06-12",
        "municipal archive",
        "text",
    ),
    (
        "history_geography",
        "Ridgeway census peak year",
        "1970",
        "census table",
        "table",
    ),
    (
        "history_geography",
        "Treaty of North Ford signing city",
        "Linden",
        "treaty index",
        "text",
    ),
    (
        "history_geography",
        "River Arno flood marker height",
        "3.7 meters",
        "survey log",
        "log",
    ),
    (
        "history_geography",
        "Old Quarter preservation zone",
        "Zone C",
        "planning map notes",
        "text",
    ),
    (
        "culture_society",
        "Festival Orion opening venue",
        "Hall 3",
        "event program",
        "text",
    ),
    (
        "culture_society",
        "Museum Night youth ticket price",
        "8 EUR",
        "ticket table",
        "table",
    ),
    (
        "culture_society",
        "Archive exhibit closing date",
        "2026-10-04",
        "exhibit calendar",
        "table",
    ),
    (
        "culture_society",
        "Civic forum moderator",
        "Amina Kroll",
        "program note",
        "text",
    ),
    (
        "culture_society",
        "Library makerspace capacity",
        "24 seats",
        "facility sheet",
        "text",
    ),
    (
        "general",
        "warehouse WH-17 emergency contact",
        "Mina Rao",
        "operations roster",
        "table",
    ),
    (
        "general",
        "support Gold Severity 1 response time",
        "15 minutes",
        "support SLA",
        "text",
    ),
    (
        "general",
        "RMA label expiry for Gold tier",
        "14 days",
        "support handbook",
        "text",
    ),
    (
        "general",
        "incident INC-730 commander",
        "Mira Chen",
        "incident table",
        "table",
    ),
    (
        "general",
        "break-glass access duration",
        "4 hours",
        "access handbook",
        "text",
    ),
    (
        "code",
        "function that applies volume discount",
        "apply_volume_discount",
        "billing_service.py",
        "code",
    ),
    (
        "code",
        "constant storing required API token environment variable",
        "FITZ_API_TOKEN",
        "config_loader.py",
        "code",
    ),
    (
        "code",
        "method that refreshes expired sessions in grace window",
        "refresh_expired_session",
        "auth_service.py",
        "code",
    ),
    (
        "code",
        "alert route for P2 search incidents",
        "route_p2_search_alert",
        "alert_router.py",
        "code",
    ),
    (
        "code",
        "export scheduler method that skips unencrypted large exports",
        "skip_unencrypted_large_export",
        "export_scheduler.py",
        "code",
    ),
)


_DISPUTED = (
    ("technology", "NimbusDB 4.2.3 ARM median write latency", "10.8 ms", "11.6 ms"),
    ("technology", "OAuthClientV2 retry backoff cap", "45 seconds", "30 seconds"),
    ("technology", "Atlas feature flag expiry", "2026-09-30", "2026-08-15"),
    ("technology", "LogStream parser error budget", "0.25 percent", "0.40 percent"),
    ("technology", "TC-4812 retry owner", "Payments Platform", "Reliability"),
    ("law_policy", "Amber board guest export rule", "cannot export", "may export after approval"),
    ("law_policy", "Cobalt CDN breach notice window", "30 days", "10 business days"),
    ("law_policy", "biometric artifact retention", "24 hours", "72 hours"),
    ("law_policy", "MeridianAI renewal notice", "75 days", "60 days"),
    ("law_policy", "service account token rotation", "45 days", "60 days"),
    ("science_medicine", "Trial Zeta cohort B median response", "18.4 percent", "21.1 percent"),
    ("science_medicine", "Lab assay L-22 storage temperature", "-80 C", "-20 C"),
    ("science_medicine", "Specimen batch RX-9 exclusion reason", "hemolysis", "label mismatch"),
    ("science_medicine", "Device D17 calibration interval", "14 days", "30 days"),
    ("science_medicine", "Study Maple enrollment target", "420 participants", "360 participants"),
    ("economics_finance", "Harborline April standard 30-year rate", "6.25 percent", "6.10 percent"),
    ("economics_finance", "Project Atlas FY27 pipeline", "7.8 million ARR", "8.4 million ARR"),
    ("economics_finance", "invoice INV-702 amount", "27100 USD", "21700 USD"),
    ("economics_finance", "Northwind CDN annual spend", "112000 USD", "98000 USD"),
    ("economics_finance", "Q4 gross margin forecast", "63.2 percent", "61.8 percent"),
    ("history_geography", "Port Selene charter date", "1894-06-12", "1895-01-09"),
    ("history_geography", "Ridgeway census peak year", "1970", "1980"),
    ("history_geography", "Treaty of North Ford signing city", "Linden", "Marrow Bay"),
    ("history_geography", "River Arno flood marker height", "3.7 meters", "4.1 meters"),
    ("history_geography", "Old Quarter preservation zone", "Zone C", "Zone B"),
    ("culture_society", "Festival Orion opening venue", "Hall 3", "Hall 4"),
    ("culture_society", "Museum Night youth ticket price", "8 EUR", "10 EUR"),
    ("culture_society", "Archive exhibit closing date", "2026-10-04", "2026-09-27"),
    ("culture_society", "Civic forum moderator", "Amina Kroll", "Jonas Veit"),
    ("culture_society", "Library makerspace capacity", "24 seats", "18 seats"),
    ("general", "warehouse WH-17 emergency contact", "Mina Rao", "Oskar Bell"),
    ("general", "support Gold Severity 1 response time", "15 minutes", "30 minutes"),
    ("general", "RMA label expiry for Gold tier", "14 days", "21 days"),
    ("general", "incident INC-730 commander", "Mira Chen", "Dana Ortiz"),
    ("general", "break-glass access duration", "4 hours", "8 hours"),
    (
        "code",
        "function that applies volume discount",
        "apply_volume_discount",
        "discounts disabled",
    ),
    (
        "code",
        "constant storing required API token environment variable",
        "FITZ_API_TOKEN",
        "FITZ_AUTH_TOKEN",
    ),
    (
        "code",
        "method that refreshes expired sessions in grace window",
        "refresh_expired_session",
        "expired sessions are never refreshed",
    ),
    (
        "code",
        "alert route for P2 search incidents",
        "route_p2_search_alert",
        "route_all_search_alerts_to_email",
    ),
    (
        "code",
        "export scheduler method that skips unencrypted large exports",
        "skip_unencrypted_large_export",
        "allow_unencrypted_large_export",
    ),
)


_INSUFFICIENT = (
    (
        "technology",
        "NimbusDB 4.2.3 ARM median write latency",
        "NimbusDB 4.2.1 x86 write latency",
        "12.4 ms",
    ),
    (
        "technology",
        "OAuthClientV2 retry backoff cap in production",
        "OAuthClientV2 staging retry cap",
        "45 seconds",
    ),
    (
        "technology",
        "Atlas feature flag expiry for EU",
        "Atlas feature flag owner",
        "Growth Platform",
    ),
    (
        "technology",
        "LogStream parser error budget for v3",
        "LogStream parser v2 error budget",
        "0.25 percent",
    ),
    (
        "technology",
        "TC-4812 retry owner after escalation",
        "TC-4812 initial reporter",
        "support desk",
    ),
    (
        "law_policy",
        "Amber board guest export rule for external auditors",
        "Amber guest comment permissions",
        "comments allowed",
    ),
    (
        "law_policy",
        "Cobalt CDN breach notice window for EU incidents",
        "Cobalt CDN uptime SLA",
        "99.95 percent",
    ),
    (
        "law_policy",
        "biometric artifact retention after appeal",
        "password reset log retention",
        "90 days",
    ),
    (
        "law_policy",
        "MeridianAI renewal notice in amended contract",
        "MeridianAI original notice",
        "75 days",
    ),
    (
        "law_policy",
        "service account token rotation for break-glass accounts",
        "standard service account rotation",
        "45 days",
    ),
    (
        "science_medicine",
        "Trial Zeta cohort B median response after week 12",
        "cohort A week 12 response",
        "18.4 percent",
    ),
    (
        "science_medicine",
        "Lab assay L-22 storage temperature during transport",
        "in-lab storage temperature",
        "-80 C",
    ),
    (
        "science_medicine",
        "Specimen batch RX-9 final exclusion reason",
        "preliminary exclusion flag",
        "review pending",
    ),
    (
        "science_medicine",
        "Device D17 calibration interval after firmware 5",
        "firmware 4 calibration interval",
        "14 days",
    ),
    (
        "science_medicine",
        "Study Maple final enrollment target",
        "draft enrollment target",
        "420 participants",
    ),
    (
        "economics_finance",
        "Harborline April standard 30-year rate with zero points",
        "one discount point rate",
        "6.25 percent",
    ),
    ("economics_finance", "Project Atlas FY28 pipeline", "FY27 pipeline", "7.8 million ARR"),
    ("economics_finance", "invoice INV-702 tax amount", "invoice INV-702 subtotal", "27100 USD"),
    (
        "economics_finance",
        "Northwind CDN annual spend after renewal",
        "pre-renewal annual spend",
        "112000 USD",
    ),
    ("economics_finance", "Q4 audited gross margin", "Q4 forecast gross margin", "63.2 percent"),
    (
        "history_geography",
        "Port Selene revised charter date",
        "original charter date",
        "1894-06-12",
    ),
    (
        "history_geography",
        "Ridgeway census peak year after annexation",
        "pre-annexation census table",
        "1970",
    ),
    ("history_geography", "Treaty of North Ford ratification city", "signing city", "Linden"),
    (
        "history_geography",
        "River Arno flood marker height at south gauge",
        "north gauge marker height",
        "3.7 meters",
    ),
    (
        "history_geography",
        "Old Quarter temporary preservation zone",
        "permanent preservation zone",
        "Zone C",
    ),
    ("culture_society", "Festival Orion closing venue", "opening venue", "Hall 3"),
    ("culture_society", "Museum Night adult ticket price", "youth ticket price", "8 EUR"),
    (
        "culture_society",
        "Archive exhibit extended closing date",
        "original closing date",
        "2026-10-04",
    ),
    ("culture_society", "Civic forum backup moderator", "main moderator", "Amina Kroll"),
    (
        "culture_society",
        "Library makerspace wheelchair-accessible capacity",
        "total capacity",
        "24 seats",
    ),
    (
        "general",
        "warehouse WH-17 night-shift emergency contact",
        "day-shift emergency contact",
        "Mina Rao",
    ),
    (
        "general",
        "support Gold Severity 0 response time",
        "Gold Severity 1 response time",
        "15 minutes",
    ),
    ("general", "RMA label expiry for Platinum tier", "Gold tier expiry", "14 days"),
    ("general", "incident INC-730 final commander", "initial commander", "Mira Chen"),
    (
        "general",
        "break-glass access duration for finance systems",
        "general break-glass duration",
        "4 hours",
    ),
    (
        "code",
        "function that applies volume discount in billing v3",
        "billing v2 method",
        "apply_volume_discount",
    ),
    (
        "code",
        "constant storing OAuth refresh token environment variable",
        "API token variable",
        "FITZ_API_TOKEN",
    ),
    (
        "code",
        "method that refreshes expired admin sessions",
        "user session refresh method",
        "refresh_expired_session",
    ),
    ("code", "alert route for P1 search incidents", "P2 search route", "route_p2_search_alert"),
    (
        "code",
        "export scheduler method that skips encrypted large exports",
        "unencrypted large export guard",
        "skip_unencrypted_large_export",
    ),
)


def build_cases() -> list[GovernanceCase]:
    """Build the deterministic 120-case balanced suite."""
    cases: list[GovernanceCase] = []
    cases.extend(_build_sufficient_cases())
    cases.extend(_build_disputed_cases())
    cases.extend(_build_insufficient_cases())
    return cases


def _build_sufficient_cases() -> list[GovernanceCase]:
    cases: list[GovernanceCase] = []
    for index, (domain, fact, answer, source, kind) in enumerate(_SUFFICIENT, start=1):
        query = f"What is the {fact}?"
        content = (
            f"{source}: For the exact requested item, {fact} is {answer}. "
            f"The row is current and explicitly scoped to the query."
        )
        cases.append(
            GovernanceCase(
                case_id=f"gov_sufficient_{index:02d}",
                domain=domain,
                query=query,
                expected_mode="sufficient",
                contexts=(_context(index, kind, source, content, "answer"),),
                tags=("governance_balanced", "sufficient", kind),
            )
        )
    return cases


def _build_disputed_cases() -> list[GovernanceCase]:
    cases: list[GovernanceCase] = []
    for index, (domain, fact, left, right) in enumerate(_DISPUTED, start=1):
        query = f"What is the {fact}?"
        left_content = (
            f"Primary source A states that {fact} is {left}. "
            "It uses the same entity, period, and scope as the query."
        )
        right_content = (
            f"Primary source B states that {fact} is {right}. "
            "It also uses the same entity, period, and scope as the query. "
            "No supersession or correction is provided."
        )
        cases.append(
            GovernanceCase(
                case_id=f"gov_disputed_{index:02d}",
                domain=domain,
                query=query,
                expected_mode="disputed",
                contexts=(
                    _context(index * 10, "text", "primary_source_a", left_content, "conflict_a"),
                    _context(
                        index * 10 + 1, "text", "primary_source_b", right_content, "conflict_b"
                    ),
                ),
                tags=("governance_balanced", "disputed", "conflict"),
            )
        )
    return cases


def _build_insufficient_cases() -> list[GovernanceCase]:
    cases: list[GovernanceCase] = []
    for index, (domain, target, near_scope, near_value) in enumerate(_INSUFFICIENT, start=1):
        query = f"What is the {target}?"
        content = (
            f"The retrieved material only covers {near_scope}: {near_value}. "
            f"It does not provide the exact requested value for {target}."
        )
        second = (
            f"A related note mentions the same general topic as {target}, "
            "but it does not contain the requested exact scope, version, period, or entity."
        )
        cases.append(
            GovernanceCase(
                case_id=f"gov_insufficient_{index:02d}",
                domain=domain,
                query=query,
                expected_mode="insufficient",
                contexts=(
                    _context(index * 20, "text", "near_miss_source", content, "near_miss"),
                    _context(index * 20 + 1, "text", "related_note", second, "related_context"),
                ),
                tags=("governance_balanced", "insufficient", "near_miss"),
            )
        )
    return cases


def _context(
    index: int,
    kind: str,
    source: str,
    content: str,
    role: str,
) -> dict[str, Any]:
    """Build one fixed evidence item dict."""
    modality = {
        "table": "table",
        "code": "code",
        "config": "configuration",
        "log": "log",
        "text": "text",
    }.get(kind, "text")
    return {
        "rank": 1 if role == "answer" else len(role),
        "source_id": f"{source}_{index}",
        "file_path": f"fixed/{source}.{kind}",
        "address_kind": "symbol" if kind == "code" else ("table" if kind == "table" else "section"),
        "address_location": source,
        "line_range": None,
        "score": 1.0,
        "excerpt": content,
        "content": content,
        "metadata": {
            "evidence_compiler": {
                "roles": [role],
                "min_sources": 2 if role.startswith("conflict") else 1,
                "content_scope": "fixed governance benchmark evidence",
                "contract": {
                    "required_modalities": [modality],
                    "source_anchors": [source],
                },
            }
        },
    }


__all__ = ["GovernanceCase", "build_cases"]
