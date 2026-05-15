# tests/unit/test_krag_multihop.py
"""
Unit tests for KragHopController.

The controller loops a RetrievalPass: run a pass -> pyrrho sufficiency
verdict -> bridge question -> run another pass. These tests mock the pass
and the governance classifier and exercise the loop control.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.retrieval.multihop import KragHopController
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from fitz_sage.governance import GovernanceDecision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_result(location: str = "mod.func", source_id: str = "src") -> ReadResult:
    """Build a ReadResult backed by a SYMBOL address."""
    addr = Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary=f"Symbol {location}",
        score=0.9,
    )
    return ReadResult(
        address=addr,
        content="def func(): return 42",
        file_path="module.py",
        line_range=(1, 3),
    )


def _make_pass(result_batches: list[list[ReadResult]]) -> MagicMock:
    """Mock RetrievalPass whose run() returns each batch in turn."""
    retrieval_pass = MagicMock(name="retrieval_pass")
    retrieval_pass.run.side_effect = result_batches
    return retrieval_pass


def _make_governance(verdicts: list[AnswerMode]) -> MagicMock:
    """Mock pyrrho classifier returning a fixed sequence of verdicts."""
    governance = MagicMock(name="governance")
    governance.decide.side_effect = [
        GovernanceDecision(mode=v, probs=(0.0, 0.0, 0.0), reason="test") for v in verdicts
    ]
    return governance


def _make_chat_factory(bridge_questions: list[str] | None = None) -> MagicMock:
    """Mock ChatFactory whose chat returns a bridge-question JSON array."""
    factory = MagicMock(name="chat_factory")
    chat = MagicMock(name="chat_client")
    chat.chat.return_value = json.dumps(bridge_questions or [])
    factory.return_value = chat
    return factory


# ---------------------------------------------------------------------------
# TestSingleHop
# ---------------------------------------------------------------------------


class TestSingleHop:
    """One pass when pyrrho judges the first hop sufficient."""

    def test_trustworthy_stops_after_one_pass(self):
        r1 = _read_result(location="mod.auth")
        rp = _make_pass([[r1]])
        gov = _make_governance([AnswerMode.TRUSTWORTHY])
        factory = _make_chat_factory()
        controller = KragHopController(
            retrieval_pass=rp, chat_factory=factory, governance=gov, max_hops=3
        )

        results = controller.execute("how does auth work?")

        assert results == [r1]
        assert rp.run.call_count == 1
        assert gov.decide.call_count == 1
        factory.assert_not_called()  # bridge extraction never reached

    def test_disputed_stops_after_one_pass(self):
        r1 = _read_result(location="mod.auth")
        rp = _make_pass([[r1]])
        controller = KragHopController(
            retrieval_pass=rp,
            chat_factory=_make_chat_factory(),
            governance=_make_governance([AnswerMode.DISPUTED]),
            max_hops=3,
        )

        results = controller.execute("query")

        assert results == [r1]
        assert rp.run.call_count == 1


# ---------------------------------------------------------------------------
# TestMultipleHops
# ---------------------------------------------------------------------------


class TestMultipleHops:
    """Bridge to a second pass when pyrrho abstains."""

    def test_abstain_bridges_to_second_pass(self):
        r1 = _read_result(location="mod.auth")
        r2 = _read_result(location="mod.session")
        rp = _make_pass([[r1], [r2]])
        controller = KragHopController(
            retrieval_pass=rp,
            chat_factory=_make_chat_factory(["what is the session handler?"]),
            governance=_make_governance([AnswerMode.ABSTAIN, AnswerMode.TRUSTWORTHY]),
            max_hops=3,
        )

        results = controller.execute("how does auth work with sessions?")

        assert results == [r1, r2]
        assert rp.run.call_count == 2

    def test_exclude_accumulates_across_hops(self):
        """Each pass is told to skip the addresses earlier hops already read."""
        r1 = _read_result(location="mod.auth", source_id="f1")
        r2 = _read_result(location="mod.session", source_id="f2")
        batches = [[r1], [r2]]
        seen_sizes: list[int] = []

        def _run(query, profile=None, *, exclude=None, **kwargs):
            seen_sizes.append(len(exclude or ()))
            return batches[len(seen_sizes) - 1]

        rp = MagicMock(name="retrieval_pass")
        rp.run.side_effect = _run

        controller = KragHopController(
            retrieval_pass=rp,
            chat_factory=_make_chat_factory(["bridge"]),
            governance=_make_governance([AnswerMode.ABSTAIN, AnswerMode.TRUSTWORTHY]),
            max_hops=3,
        )
        controller.execute("query")

        # Hop 1 excludes nothing; hop 2 excludes hop 1's one result.
        assert seen_sizes == [0, 1]


# ---------------------------------------------------------------------------
# TestStopConditions
# ---------------------------------------------------------------------------


class TestStopConditions:
    """Loop termination conditions."""

    def test_stops_when_pass_returns_empty(self):
        gov = _make_governance([])
        controller = KragHopController(
            retrieval_pass=_make_pass([[]]),
            chat_factory=_make_chat_factory(),
            governance=gov,
            max_hops=3,
        )

        results = controller.execute("query about nothing")

        assert results == []
        gov.decide.assert_not_called()

    def test_stops_at_max_hops(self):
        r1 = _read_result(location="mod.a")
        r2 = _read_result(location="mod.b")
        rp = _make_pass([[r1], [r2]])
        controller = KragHopController(
            retrieval_pass=rp,
            chat_factory=_make_chat_factory(["next question"]),
            governance=_make_governance([AnswerMode.ABSTAIN, AnswerMode.ABSTAIN]),
            max_hops=2,
        )

        results = controller.execute("complex query")

        assert results == [r1, r2]
        assert rp.run.call_count == 2

    def test_stops_when_no_bridge_questions(self):
        r1 = _read_result(location="mod.func")
        rp = _make_pass([[r1]])
        controller = KragHopController(
            retrieval_pass=rp,
            chat_factory=_make_chat_factory([]),  # no bridge questions
            governance=_make_governance([AnswerMode.ABSTAIN]),
            max_hops=5,
        )

        results = controller.execute("query")

        assert results == [r1]
        assert rp.run.call_count == 1

    def test_no_governance_loops_to_max_hops(self):
        """With governance disabled there is no sufficiency signal — the loop
        runs to max_hops (or until bridge extraction dries up)."""
        r1 = _read_result(location="mod.a")
        r2 = _read_result(location="mod.b")
        rp = _make_pass([[r1], [r2]])
        controller = KragHopController(
            retrieval_pass=rp,
            chat_factory=_make_chat_factory(["keep going"]),
            governance=None,
            max_hops=2,
        )

        results = controller.execute("query")

        assert results == [r1, r2]
        assert rp.run.call_count == 2


# ---------------------------------------------------------------------------
# TestProfileForwarding
# ---------------------------------------------------------------------------


class TestProfileForwarding:
    """The controller forwards the retrieval profile to each pass."""

    def test_forwards_profile_to_pass(self):
        from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile

        rp = _make_pass([[_read_result()]])
        profile = RetrievalProfile()
        controller = KragHopController(
            retrieval_pass=rp,
            chat_factory=_make_chat_factory(),
            governance=_make_governance([AnswerMode.TRUSTWORTHY]),
            max_hops=2,
        )

        controller.execute("query", profile)

        rp.run.assert_called_once()
        assert rp.run.call_args[0][1] is profile
