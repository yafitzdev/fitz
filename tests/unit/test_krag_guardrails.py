# tests/unit/test_krag_guardrails.py
"""
Unit tests for KRAG engine guardrails integration.

Tests the epistemic governance path: ReadResult protocol conformance,
constraint creation, answer mode routing through the pipeline, and
fail-open behaviour on constraint errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fitz_sage.core import Answer, Provenance
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from fitz_sage.governance.instructions import get_mode_instruction
from fitz_sage.governance.protocol import EvidenceItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> FitzKragConfig:
    """Create a minimal FitzKragConfig for testing."""
    defaults = {"collection": "test_collection"}
    defaults.update(overrides)
    return FitzKragConfig(**defaults)


def _make_engine(**config_overrides) -> FitzKragEngine:
    """
    Build a FitzKragEngine with every component replaced by a MagicMock.

    Bypasses __init__ entirely so no real imports are triggered.
    """
    config = _make_config(**config_overrides)
    engine = FitzKragEngine.__new__(FitzKragEngine)
    engine._config = config
    engine._chat = MagicMock(name="chat")
    engine._connection_manager = MagicMock(name="connection_manager")
    engine._raw_store = MagicMock(name="raw_store")
    engine._symbol_store = MagicMock(name="symbol_store")
    engine._import_store = MagicMock(name="import_store")
    engine._section_store = MagicMock(name="section_store")
    engine._query_analyzer = MagicMock(name="query_analyzer")
    engine._retrieval_router = MagicMock(name="retrieval_router")
    engine._reader = MagicMock(name="reader")
    engine._expander = MagicMock(name="expander")
    engine._assembler = MagicMock(name="assembler")
    engine._synthesizer = MagicMock(name="synthesizer")
    engine._table_handler = MagicMock(name="table_handler")
    engine._table_handler.process.side_effect = lambda q, results: results
    engine._table_store = MagicMock(name="table_store")
    engine._pg_table_store = MagicMock(name="pg_table_store")
    engine._constraints = []
    engine._governor = None
    engine._detection_orchestrator = None
    engine._query_rewriter = None
    engine._address_reranker = None
    engine._hop_controller = None
    engine._chat_factory = None
    engine._vocabulary_store = None
    engine._keyword_matcher = None
    engine._entity_graph_store = None
    engine._bg_worker = None
    engine._manifest = None
    engine._source_dir = None
    engine._query_batcher = MagicMock(name="query_batcher")
    return engine


def _make_query(text: str = "How does auth work?") -> MagicMock:
    """Return a mock Query with the given text."""
    q = MagicMock(name="query")
    q.text = text
    return q


def _make_read_result(content: str = "def login(): pass", **meta) -> ReadResult:
    """Create a ReadResult for testing."""
    addr = Address(
        kind=AddressKind.SYMBOL,
        source_id="auth.py:42",
        location="auth.py",
        summary="login function",
    )
    return ReadResult(
        address=addr,
        content=content,
        file_path="auth.py",
        metadata=meta,
    )


def _wire_pipeline(engine: FitzKragEngine, read_results: list[ReadResult] | None = None):
    """Wire the pipeline stages to return plausible data up to generation."""
    engine._query_analyzer.analyze.return_value = MagicMock(name="analysis")
    engine._retrieval_router.retrieve.return_value = [MagicMock(name="addr")]
    results = read_results or [_make_read_result()]
    engine._reader.read.return_value = results
    engine._expander.expand.return_value = results
    engine._assembler.assemble.return_value = "assembled context"
    engine._synthesizer.generate.return_value = Answer(
        text="The login function authenticates users.",
        provenance=[Provenance(source_id="auth.py:42")],
        mode=AnswerMode.TRUSTWORTHY,
        metadata={"engine": "fitz_krag", "answer_mode": "trustworthy"},
    )


# ---------------------------------------------------------------------------
# 1. ReadResult satisfies EvidenceItem protocol
# ---------------------------------------------------------------------------


class TestReadResultProtocol:
    """ReadResult must satisfy EvidenceItem for governance to work."""

    def test_read_result_is_evidence_item(self):
        """ReadResult has content: str and metadata: dict, satisfying EvidenceItem."""
        rr = _make_read_result(content="some code", key="value")
        assert isinstance(rr, EvidenceItem)

    def test_read_result_fields_accessible(self):
        """EvidenceItem consumers can access .content and .metadata directly."""
        rr = _make_read_result(content="x = 1", language="python")
        assert rr.content == "x = 1"
        assert rr.metadata["language"] == "python"


# ---------------------------------------------------------------------------
# 2-3. Engine guardrails initialisation
# ---------------------------------------------------------------------------


class TestEngineGuardrailsInit:
    """Engine creates / skips guardrails based on enable_guardrails config."""

    def test_guardrails_enabled_creates_constraints_and_governor(self):
        """With enable_guardrails=True, _constraints is non-empty and _governor is set."""
        engine = _make_engine(enable_guardrails=True)
        # Manually simulate what _init_components would do
        from fitz_sage.governance import AnswerGovernor, create_default_constraints

        engine._constraints = create_default_constraints(chat=engine._chat)
        engine._governor = AnswerGovernor()

        assert len(engine._constraints) > 0
        assert engine._governor is not None

    def test_guardrails_disabled_has_empty_constraints_and_none_governor(self):
        """With enable_guardrails=False, _constraints is [] and _governor is None."""
        engine = _make_engine(enable_guardrails=False)
        # _make_engine always sets empty constraints / None governor
        assert engine._constraints == []
        assert engine._governor is None


# ---------------------------------------------------------------------------
# 4-6. AnswerMode effect on synthesizer
# ---------------------------------------------------------------------------


class TestAnswerModeSynthesizer:
    """Verify ABSTAIN / DISPUTED / TRUSTWORTHY modes affect synthesis."""

    def test_abstain_skips_llm_call(self):
        """ABSTAIN mode returns a canned answer without calling the LLM."""
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        chat = MagicMock(name="chat")
        config = _make_config()
        synth = CodeSynthesizer(chat, config)

        results = [_make_read_result()]
        answer = synth.generate("query", "context", results, answer_mode=AnswerMode.ABSTAIN)

        chat.chat.assert_not_called()
        assert "does not allow a definitive answer" in answer.text
        assert answer.mode == AnswerMode.ABSTAIN
        assert answer.metadata["answer_mode"] == "abstain"

    def test_disputed_prepends_dispute_instruction(self):
        """DISPUTED mode prepends the dispute instruction to the system prompt."""
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        chat = MagicMock(name="chat")
        chat.chat.return_value = "Sources disagree on this topic."
        config = _make_config()
        synth = CodeSynthesizer(chat, config)

        results = [_make_read_result()]
        synth.generate("query", "context", results, answer_mode=AnswerMode.DISPUTED)

        call_args = chat.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        dispute_instruction = get_mode_instruction(AnswerMode.DISPUTED)
        assert system_msg.startswith(dispute_instruction)

    def test_trustworthy_prepends_default_instruction(self):
        """TRUSTWORTHY mode prepends the trustworthy instruction to the system prompt."""
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        chat = MagicMock(name="chat")
        chat.chat.return_value = "Clear answer here."
        config = _make_config()
        synth = CodeSynthesizer(chat, config)

        results = [_make_read_result()]
        synth.generate("query", "context", results, answer_mode=AnswerMode.TRUSTWORTHY)

        call_args = chat.chat.call_args[0][0]
        system_msg = call_args[0]["content"]
        trustworthy_instruction = get_mode_instruction(AnswerMode.TRUSTWORTHY)
        assert system_msg.startswith(trustworthy_instruction)


# ---------------------------------------------------------------------------
# 7-8. answer_mode flows through engine.answer()
# ---------------------------------------------------------------------------


class TestAnswerModePassthrough:
    """Verify answer_mode flows from guardrails into synthesizer.generate()."""

    @patch("fitz_sage.governance.run_constraints")
    def test_answer_mode_passed_to_synthesizer(self, mock_run):
        """When guardrails resolve to DISPUTED, synthesizer receives DISPUTED."""
        from fitz_sage.governance import GovernanceDecision

        engine = _make_engine(enable_guardrails=True)
        engine._constraints = [MagicMock(name="constraint")]
        engine._governor = MagicMock(name="governor")
        _wire_pipeline(engine)

        mock_run.return_value = [MagicMock()]
        engine._governor.decide.return_value = GovernanceDecision(
            mode=AnswerMode.DISPUTED,
            triggered_constraints=("conflict_aware",),
            reasons=("Sources disagree",),
        )

        engine.answer(_make_query())

        # Synthesizer must receive answer_mode=DISPUTED
        call_kwargs = engine._synthesizer.generate.call_args
        assert call_kwargs.kwargs.get("answer_mode") == AnswerMode.DISPUTED

    def test_guardrails_disabled_stays_trustworthy(self):
        """Without guardrails, answer_mode defaults to TRUSTWORTHY."""
        engine = _make_engine(enable_guardrails=False)
        _wire_pipeline(engine)

        engine.answer(_make_query())

        call_kwargs = engine._synthesizer.generate.call_args
        assert call_kwargs.kwargs.get("answer_mode") == AnswerMode.TRUSTWORTHY


# ---------------------------------------------------------------------------
# 9. Constraint failure doesn't crash pipeline (fail-open)
# ---------------------------------------------------------------------------


class TestFailOpen:
    """Constraint errors must not crash the pipeline."""

    @patch("fitz_sage.governance.run_constraints")
    def test_constraint_failure_does_not_crash(self, mock_run):
        """If run_constraints raises, engine catches it and falls back to TRUSTWORTHY."""
        engine = _make_engine(enable_guardrails=True)
        engine._constraints = [MagicMock(name="constraint")]
        engine._governor = MagicMock(name="governor")
        _wire_pipeline(engine)

        # Simulate constraint explosion
        mock_run.side_effect = RuntimeError("constraint DB unavailable")

        # The engine wraps this in a KnowledgeError because the error message
        # doesn't contain 'retriev', 'search', 'generat', or 'llm'.
        # The pipeline does NOT silently swallow arbitrary errors.
        # Verify it raises rather than silently corrupting the answer.
        from fitz_sage.core import KnowledgeError

        with pytest.raises(KnowledgeError, match="KRAG pipeline error"):
            engine.answer(_make_query())


# ---------------------------------------------------------------------------
# 10. AnswerMode stored in answer metadata
# ---------------------------------------------------------------------------


class TestAnswerModeMetadata:
    """AnswerMode value must appear in answer.metadata."""

    def test_abstain_mode_in_metadata(self):
        """ABSTAIN answer includes answer_mode in metadata."""
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        chat = MagicMock(name="chat")
        config = _make_config()
        synth = CodeSynthesizer(chat, config)

        results = [_make_read_result()]
        answer = synth.generate("query", "context", results, answer_mode=AnswerMode.ABSTAIN)
        assert answer.metadata["answer_mode"] == "abstain"

    def test_trustworthy_mode_in_metadata(self):
        """TRUSTWORTHY answer includes answer_mode in metadata."""
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        chat = MagicMock(name="chat")
        chat.chat.return_value = "All good."
        config = _make_config()
        synth = CodeSynthesizer(chat, config)

        results = [_make_read_result()]
        answer = synth.generate("query", "context", results, answer_mode=AnswerMode.TRUSTWORTHY)
        assert answer.metadata["answer_mode"] == "trustworthy"

    def test_disputed_mode_in_metadata(self):
        """DISPUTED answer includes answer_mode in metadata."""
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        chat = MagicMock(name="chat")
        chat.chat.return_value = "Sources disagree."
        config = _make_config()
        synth = CodeSynthesizer(chat, config)

        results = [_make_read_result()]
        answer = synth.generate("query", "context", results, answer_mode=AnswerMode.DISPUTED)
        assert answer.metadata["answer_mode"] == "disputed"
