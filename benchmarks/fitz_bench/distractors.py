"""Deterministic near-neighbor documents for corpus-growth testing."""

from __future__ import annotations

import shutil
from pathlib import Path

_TOPICS = (
    "Access and Security",
    "Billing and Invoices",
    "Customer Support",
    "Deployment Operations",
    "Finance Forecast",
    "Incident Review",
    "Procurement and Vendors",
    "Product Rollout",
    "Service Ownership",
    "Warehouse Operations",
)
_REGIONS = ("APAC", "EMEA", "North America", "LATAM")
_TEAMS = ("Amber", "Cobalt", "Juniper", "Quartz", "Willow")


def stage_corpus(source: Path, target: Path, *, distractors: int) -> Path:
    """Copy a corpus and add deterministic company-document distractors."""
    if target.exists():
        shutil.rmtree(target)
    if source.is_file():
        target.mkdir(parents=True)
        shutil.copy2(source, target / source.name)
    else:
        shutil.copytree(source, target)

    noise_dir = target / "_distractors"
    noise_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, distractors + 1):
        topic = _TOPICS[(index - 1) % len(_TOPICS)]
        region = _REGIONS[(index - 1) % len(_REGIONS)]
        team = _TEAMS[(index - 1) % len(_TEAMS)]
        identifier = f"ARCH-{7000 + index}"
        content = _document(
            index=index,
            topic=topic,
            region=region,
            team=team,
            identifier=identifier,
        )
        (noise_dir / f"archive_{index:04d}.md").write_text(content, encoding="utf-8")
    return target


def _document(*, index: int, topic: str, region: str, team: str, identifier: str) -> str:
    year = 2018 + (index % 5)
    return "\n".join(
        [
            f"# Archived {topic} Brief {identifier}",
            "",
            f"This historical {year} record covers {topic.lower()} for the {region} region.",
            (
                f"The retired {team} team owned the review. This document is retained for "
                "audit history and is not the current policy or source of truth."
            ),
            "",
            "## Operational Notes",
            "",
            (
                "The review mentions customer requests, service owners, response targets, "
                "security controls, budgets, project status, deployment checks, and incident "
                "follow-up. Values in this archive apply only to its retired program."
            ),
            "",
            "## Record Control",
            "",
            f"Archive identifier: {identifier}. Superseded in {year + 1}.",
            "",
        ]
    )


__all__ = ["stage_corpus"]
