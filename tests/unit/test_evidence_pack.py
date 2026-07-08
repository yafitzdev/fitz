# tests/unit/test_evidence_pack.py
"""Tests for retrieval-first evidence contracts."""

from __future__ import annotations

import json

from fitz_sage.core import EvidenceItem, EvidencePack
from fitz_sage.core.answer_mode import AnswerMode


def test_evidence_pack_json_round_trips():
    """Evidence packs serialize to plain JSON-compatible data."""
    pack = EvidencePack(
        query="Which test failed?",
        mode=AnswerMode.SUFFICIENT,
        items=[
            EvidenceItem(
                rank=1,
                source_id="doc-1",
                file_path="docs/sprint.md",
                address_kind="section",
                address_location="Sprint 47",
                line_range=(12, 18),
                score=0.87,
                excerpt="Sprint 47 failed in payment retry tests.",
                content="Sprint 47 failed in payment retry tests.",
                metadata={"kind": AnswerMode.SUFFICIENT},
            )
        ],
        reasons=["Sources support the answer."],
        timings={"Retrieval": 0.12},
        indexing_status={"complete": True},
    )

    data = json.loads(pack.to_json())

    assert data["mode"] == "sufficient"
    assert data["items"][0]["line_range"] == [12, 18]
    assert data["items"][0]["metadata"]["kind"] == "sufficient"
