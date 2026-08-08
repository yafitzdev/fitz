"""Tests for mechanical progressive evidence delivery."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fitz_sage.engines.fitz_krag.evidence_delivery import deliver_progressively
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult


def _results(count: int) -> list[ReadResult]:
    return [
        ReadResult(
            address=Address(
                kind=AddressKind.SECTION,
                source_id=f"doc-{index}",
                location=f"Section {index}",
                summary=f"Section {index}",
                score=1.0 - (index * 0.01),
            ),
            content=f"Evidence {index}",
            file_path=f"docs/{index}.md",
            line_range=(index, index),
        )
        for index in range(1, count + 1)
    ]


def _decision(verdict: str, reason: str = "model reason") -> SimpleNamespace:
    payload = {
        "schema_version": 1,
        "verdict": verdict,
        "reason": reason,
    }
    return SimpleNamespace(
        verdict=verdict,
        reasons=(reason,),
        to_dict=lambda: dict(payload),
    )


class _Runtime:
    def __init__(self, decisions: list[SimpleNamespace]) -> None:
        self._decisions = iter(decisions)
        self.evidence_sets: list[list[dict[str, str]]] = []

    def decide(self, query: str, evidence: list[dict[str, str]]) -> SimpleNamespace:
        assert query == "question"
        self.evidence_sets.append(evidence)
        return next(self._decisions)


def test_starts_with_three_then_adds_two_until_sufficient() -> None:
    insufficient = _decision("INSUFFICIENT", "Need more evidence.")
    sufficient = _decision("SUFFICIENT", "Enough evidence.")
    runtime = _Runtime([insufficient, sufficient])

    delivery = deliver_progressively(runtime, "question", _results(7))

    assert len(delivery.selected) == 5
    assert delivery.decision is sufficient
    assert [len(evidence) for evidence in runtime.evidence_sets] == [3, 5]
    assert [[item["source_id"] for item in evidence] for evidence in runtime.evidence_sets] == [
        ["doc-1", "doc-2", "doc-3"],
        ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"],
    ]
    assert delivery.metadata(available=7, limit=7) == {
        "available": 7,
        "selected": 5,
        "limit": 7,
        "initial_prefix_size": 3,
        "prefix_increment": 2,
        "evaluated_prefixes": [3, 5],
        "trajectory": [
            {"evidence_count": 3, "decision": insufficient.to_dict()},
            {"evidence_count": 5, "decision": sufficient.to_dict()},
        ],
    }


def test_disputed_is_terminal_without_local_override() -> None:
    disputed = _decision("DISPUTED", "Sources conflict.")
    runtime = _Runtime([disputed])

    delivery = deliver_progressively(runtime, "question", _results(8))

    assert len(delivery.selected) == 3
    assert delivery.decision is disputed
    assert [len(evidence) for evidence in runtime.evidence_sets] == [3]


def test_insufficient_grows_through_short_final_prefix() -> None:
    decisions = [_decision("INSUFFICIENT", f"Attempt {index}") for index in range(4)]
    runtime = _Runtime(decisions)

    delivery = deliver_progressively(runtime, "question", _results(8))

    assert len(delivery.selected) == 8
    assert delivery.decision is decisions[-1]
    assert [len(evidence) for evidence in runtime.evidence_sets] == [3, 5, 7, 8]


@pytest.mark.parametrize("count", [1, 2])
def test_small_result_sets_are_evaluated_once(count: int) -> None:
    decision = _decision("INSUFFICIENT")
    runtime = _Runtime([decision])

    delivery = deliver_progressively(runtime, "question", _results(count))

    assert len(delivery.selected) == count
    assert [len(evidence) for evidence in runtime.evidence_sets] == [count]


def test_empty_result_set_still_uses_pyrrhos_exact_empty_decision() -> None:
    decision = _decision("INSUFFICIENT", "No evidence.")
    runtime = _Runtime([decision])

    delivery = deliver_progressively(runtime, "question", [])

    assert delivery.selected == ()
    assert delivery.decision is decision
    assert runtime.evidence_sets == [[]]
