# tests/unit/test_krag_engine.py
"""
Unit tests for FitzKragEngine.

All dependencies are mocked. The engine is constructed via __new__ with
mocked internals for answer/ingest/config tests. Only the init tests
exercise the real __init__ with patched imports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from fitz_sage.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Provenance,
    Query,
    QueryError,
)
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine, _build_provider_config
from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from tests.unit.mock_engine import build_mock_engine

# Tests in this file @patch SqliteConnectionManager. Without resetting the
# singleton, the MagicMock leaks into subsequent tests in collection order
# and cascades into ~25 failures across test_vocabulary / test_section_store /
# test_krag_guardrails. The fixture is defined in tests/unit/conftest.py.
pytestmark = pytest.mark.usefixtures("reset_sqlite_singleton")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides) -> FitzKragConfig:
    """Create a minimal FitzKragConfig for testing."""
    defaults = {"collection": "test_collection"}
    defaults.update(overrides)
    return FitzKragConfig(**defaults)


# Mock engine builder is shared across test_krag_{detection,engine,query_rewriting}.
_make_engine = build_mock_engine


def _make_query(text: str = "How does auth work?") -> MagicMock:
    """Return a mock Query with the given text."""
    q = MagicMock(name="query")
    q.text = text
    return q


def _decision(
    mode: AnswerMode,
    reason: str,
    *,
    probs: tuple[float, float, float] | None = None,
) -> MagicMock:
    """Build a lightweight governance decision for cutoff tests."""
    decision = MagicMock()
    decision.mode = mode
    decision.reasons = (reason,)
    decision.reason = reason
    if probs is not None:
        decision.probs = probs
    elif mode is AnswerMode.SUFFICIENT:
        decision.probs = (0.11, 0.22, 0.67)
    elif mode is AnswerMode.DISPUTED:
        decision.probs = (0.12, 0.68, 0.20)
    else:
        decision.probs = (0.69, 0.18, 0.13)
    return decision


def _evidence_results(count: int) -> tuple[list[Address], list[ReadResult]]:
    """Build ranked address/read-result fixtures for evidence tests."""
    addresses = [
        Address(
            kind=AddressKind.SECTION,
            source_id=f"doc-{i}",
            location=f"Section {i}",
            summary=f"Section {i}",
            score=1.0 - (i * 0.1),
        )
        for i in range(1, count + 1)
    ]
    results = [
        ReadResult(
            address=address,
            content=f"Evidence body {i}",
            file_path=f"docs/{i}.md",
            line_range=(i, i),
        )
        for i, address in enumerate(addresses, start=1)
    ]
    return addresses, results


# ---------------------------------------------------------------------------
# TestEngineInit
# ---------------------------------------------------------------------------


class TestEngineInit:
    """Tests that exercise the real __init__ / _init_components path."""

    # All lazy imports inside _init_components must be patched at the
    # location where they are imported (the engine module's namespace).
    _ENGINE_MOD = "fitz_sage.engines.fitz_krag.engine"

    @pytest.fixture()
    def _patches(self):
        """
        Context-manager that patches every lazy import used by
        _init_components and yields a dict of the mock objects.
        """
        names = {
            # llm
            "get_chat": "fitz_sage.llm.client.get_chat",
            # storage
            "SqliteConnectionManager": ("fitz_sage.storage.sqlite.SqliteConnectionManager"),
            # stores
            "RawFileStore": ("fitz_sage.engines.fitz_krag.ingestion.raw_file_store.RawFileStore"),
            "SymbolStore": ("fitz_sage.engines.fitz_krag.ingestion.symbol_store.SymbolStore"),
            "ImportGraphStore": (
                "fitz_sage.engines.fitz_krag.ingestion.import_graph_store.ImportGraphStore"
            ),
            "SectionStore": ("fitz_sage.engines.fitz_krag.ingestion.section_store.SectionStore"),
            "TableStoreKrag": ("fitz_sage.engines.fitz_krag.ingestion.table_store.TableStore"),
            "ensure_schema": ("fitz_sage.engines.fitz_krag.ingestion.schema.ensure_schema"),
            # strategies
            "CodeSearchStrategy": (
                "fitz_sage.engines.fitz_krag.retrieval.strategies.code_search.CodeSearchStrategy"
            ),
            "SectionSearchStrategy": (
                "fitz_sage.engines.fitz_krag.retrieval"
                ".strategies.section_search.SectionSearchStrategy"
            ),
            "TableSearchStrategy": (
                "fitz_sage.engines.fitz_krag.retrieval.strategies.table_search.TableSearchStrategy"
            ),
            # retrieval
            "RetrievalRouter": ("fitz_sage.engines.fitz_krag.retrieval.router.RetrievalRouter"),
            "ContentReader": ("fitz_sage.engines.fitz_krag.retrieval.reader.ContentReader"),
            "CodeExpander": ("fitz_sage.engines.fitz_krag.retrieval.expander.CodeExpander"),
            "TableQueryHandler": (
                "fitz_sage.engines.fitz_krag.retrieval.table_handler.TableQueryHandler"
            ),
            # context + generation
            "ContextAssembler": ("fitz_sage.engines.fitz_krag.context.assembler.ContextAssembler"),
            "CodeSynthesizer": (
                "fitz_sage.engines.fitz_krag.generation.synthesizer.CodeSynthesizer"
            ),
            # shared tabular (SQLite-backed after Cloud removal)
            "SqliteTableStore": "fitz_sage.tabular.store.sqlite.SqliteTableStore",
            # factory
            "get_chat_factory": "fitz_sage.llm.factory.get_chat_factory",
        }

        patchers = {key: patch(target) for key, target in names.items()}
        mocks = {}
        for key, p in patchers.items():
            mocks[key] = p.start()

        # SqliteConnectionManager.get_instance() returns a mock
        mocks["SqliteConnectionManager"].get_instance.return_value = MagicMock(name="pg_instance")

        yield mocks

        for p in patchers.values():
            p.stop()

    def test_init_creates_components(self, _patches):
        """Engine initialises without error when all deps are available."""
        config = _make_config()
        engine = FitzKragEngine(config)

        # Required enrichment uses the managed ONNX provider directly.
        _patches["get_chat"].assert_not_called()
        assert engine._enricher_chat is engine._summarizer_chat
        _patches["SqliteConnectionManager"].get_instance.assert_called()

        # Schema ensured
        _patches["ensure_schema"].assert_called_once()

        # Strategies and router created
        _patches["CodeSearchStrategy"].assert_called_once()
        _patches["SectionSearchStrategy"].assert_called_once()
        _patches["TableSearchStrategy"].assert_called_once()
        _patches["RetrievalRouter"].assert_called_once()

        # Reader, expander, assembler, table handler created
        _patches["ContentReader"].assert_called_once()
        _patches["CodeExpander"].assert_called_once()
        _patches["TableQueryHandler"].assert_called_once()
        _patches["ContextAssembler"].assert_called_once()
        _patches["CodeSynthesizer"].assert_not_called()

        # Config stored correctly
        assert engine.config is config

    def test_init_creates_synthesizer_when_configured(self, _patches):
        """A synthesizer provider creates the answer generator explicitly."""
        config = _make_config(synthesizer="endpoint/qwen2.5-7b-instruct")

        FitzKragEngine(config)

        _patches["get_chat"].assert_has_calls(
            [
                call(
                    "endpoint/qwen2.5-7b-instruct",
                    "smart",
                    {"base_url": "http://127.0.0.1:8080/v1"},
                ),
            ]
        )
        assert _patches["get_chat"].call_count == 1
        _patches["CodeSynthesizer"].assert_called_once()

    def test_provider_config_forwards_auth_block(self):
        """Role providers receive auth config without needing chat tiers."""
        auth = {
            "type": "enterprise",
            "token_url": "https://auth.example.com/token",
            "client_id": "${CLIENT_ID}",
            "client_secret": "${CLIENT_SECRET}",
            "llm_api_key_env": "CORP_LLM_API_KEY",
        }

        config = _build_provider_config(
            "https://llm.corp.internal/v1",
            None,
            spec="enterprise/openai/gpt-4o",
            auth=auth,
            cert_path="/etc/ssl/corp-ca-bundle.crt",
        )

        assert config == {
            "base_url": "https://llm.corp.internal/v1",
            "cert_path": "/etc/ssl/corp-ca-bundle.crt",
            "auth": auth,
        }

    def test_provider_config_merges_api_key_env_into_auth_block(self):
        """Endpoint auth can keep custom header options from YAML."""
        config = _build_provider_config(
            "https://api.together.xyz/v1",
            "TOGETHER_API_KEY",
            spec="endpoint/meta-llama-3.1-70b",
            auth={"header_format": "bearer"},
        )

        assert config == {
            "base_url": "https://api.together.xyz/v1",
            "auth": {
                "header_format": "bearer",
                "api_key_env": "TOGETHER_API_KEY",
            },
        }

    def test_default_config_has_no_chat_tier_factory(self):
        """Retrieval-only defaults should not build a tiered chat factory."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config()

        assert engine._chat_tier_specs() is None

    def test_configured_chat_tiers_build_factory_specs(self):
        """Configured chat tiers still build complete factory specs."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(chat_fast="endpoint/qwen2.5-7b")

        assert engine._chat_tier_specs() == {
            "fast": "endpoint/qwen2.5-7b",
            "balanced": "endpoint/qwen2.5-7b",
            "smart": "endpoint/qwen2.5-7b",
        }

    def test_init_failure_raises_configuration_error(self):
        """
        If _init_components raises, __init__ wraps it as
        ConfigurationError.
        """
        with patch.object(
            FitzKragEngine,
            "_init_components",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(ConfigurationError, match="boom"):
                FitzKragEngine(_make_config())


# ---------------------------------------------------------------------------
# TestAnswer
# ---------------------------------------------------------------------------


class TestAnswer:
    """Tests for the answer() pipeline."""

    def test_answer_full_flow(self):
        """Happy path: every stage returns valid data."""
        engine = _make_engine()
        query = _make_query(
            "What does the login function do when the user provides invalid credentials?"
        )

        # Wire up the pipeline stages
        address_1 = MagicMock(name="addr1")
        address_2 = MagicMock(name="addr2")
        engine._retrieval_router.retrieve.return_value = [
            address_1,
            address_2,
        ]

        read_1 = MagicMock(name="read1")
        engine._reader.read.return_value = [read_1]

        expanded = [MagicMock(name="expanded1")]
        engine._expander.expand.return_value = expanded

        # Table handler passes through (side_effect from _make_engine)

        context = MagicMock(name="context")
        engine._assembler.assemble.return_value = context

        expected_answer = Answer(
            text="The login function authenticates users.",
            provenance=[Provenance(source_id="auth.py:42")],
            metadata={"engine": "fitz_krag"},
        )
        engine._synthesizer.generate.return_value = expected_answer

        # Execute
        result = engine.answer(query)

        # Verify each stage called with correct args
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0][0] == query.text
        from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile

        assert isinstance(call_args[0][1], RetrievalProfile)
        assert call_args[1]["rewrite_result"] is None
        engine._reader.read.assert_called_once_with(
            [address_1, address_2],
            engine._config.top_read,
        )
        engine._expander.expand.assert_called_once_with([read_1], entity_expansion_limit=3)
        engine._table_handler.process.assert_called_once_with(
            query.text,
            expanded,
            allow_sql_generation=True,
        )
        engine._assembler.assemble.assert_called_once_with(
            query.text,
            expanded,
        )
        from fitz_sage.core.answer_mode import AnswerMode

        engine._synthesizer.generate.assert_called_once_with(
            query.text,
            context,
            expanded,
            answer_mode=AnswerMode.SUFFICIENT,
            gap_context=None,
            conflict_context=None,
        )

        assert result is expected_answer
        assert result.metadata["pyrrho"]["mode"] == "sufficient"
        assert result.metadata["pyrrho"]["probabilities"] == {
            "insufficient": 0.1,
            "disputed": 0.1,
            "sufficient": 0.8,
        }
        assert result.metadata["query_profile"]["profile"]["top_k"] == engine._config.top_addresses

    def test_answer_uses_no_chat_query_planner_by_default(self):
        """Default query prep is deterministic unless query_intelligence is configured."""
        engine = _make_engine()
        query = _make_query("Compare Q1 2024 vs Q2 2024 API failures")
        engine._query_batcher.batch_classify.side_effect = AssertionError("chat prep called")

        address = MagicMock(name="addr")
        engine._retrieval_router.retrieve.return_value = [address]
        read_result = MagicMock(name="read")
        engine._reader.read.return_value = [read_result]
        engine._expander.expand.return_value = [read_result]
        engine._assembler.assemble.return_value = MagicMock()
        expected = Answer(text="Answer.", provenance=[], metadata={})
        engine._synthesizer.generate.return_value = expected

        result = engine.answer(query)

        assert result is expected
        engine._query_batcher.batch_classify.assert_not_called()
        call_args = engine._retrieval_router.retrieve.call_args
        profile = call_args[0][1]
        assert "q1 2024" in profile.temporal_references
        assert "q2 2024" in profile.temporal_references
        assert profile.planning_owner == "pyrrho"
        engine._governance.plan_query.assert_called_once_with(query.text)

    def test_answer_empty_query_raises(self):
        """Empty or whitespace-only query text raises QueryError."""
        engine = _make_engine()

        for blank in ("", "   ", "\t\n"):
            q = _make_query(blank)
            with pytest.raises(QueryError, match="empty"):
                engine.answer(q)

    def test_answer_without_synthesizer_raises_actionable_error(self):
        """Answer mode requires explicit synthesis configuration."""
        engine = _make_engine()
        engine._synthesizer = None

        with pytest.raises(GenerationError, match="No synthesizer configured"):
            engine.answer(_make_query("What is RAG?"))

    def test_answer_no_addresses_returns_fallback(self):
        """Router returning [] yields an actionable INSUFFICIENT answer."""
        engine = _make_engine()
        query = _make_query()

        engine._retrieval_router.retrieve.return_value = []

        result = engine.answer(query)

        assert result.provenance == []
        assert result.mode == AnswerMode.INSUFFICIENT
        assert result.metadata["engine"] == "fitz_krag"
        assert result.metadata["query"] == query.text
        assert result.metadata["answer_mode"] == "insufficient"
        assert "gap_context" in result.metadata

        # Reader should never be called
        engine._reader.read.assert_not_called()

    def test_answer_no_read_results_returns_fallback(self):
        """Retrieval finding addresses but reading nothing yields an insufficient Answer."""
        engine = _make_engine()
        query = _make_query()

        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = []

        result = engine.answer(query)

        assert result.mode == AnswerMode.INSUFFICIENT
        assert result.metadata["answer_mode"] == "insufficient"
        assert "gap_context" in result.metadata
        assert result.provenance == []

        # Expander should never be called
        engine._expander.expand.assert_not_called()

    def test_answer_retrieval_error_raises_knowledge_error(self):
        """
        When the retrieval router raises an error whose message
        contains 'search', it is wrapped as KnowledgeError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._retrieval_router.retrieve.side_effect = RuntimeError(
            "vector search connection timeout"
        )

        with pytest.raises(KnowledgeError, match="Retrieval failed"):
            engine.answer(query)

    def test_answer_retrieval_error_with_retriev_keyword(self):
        """
        Error message containing 'retriev' also maps to
        KnowledgeError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._retrieval_router.retrieve.side_effect = RuntimeError("retrieval timeout")

        with pytest.raises(KnowledgeError, match="Retrieval failed"):
            engine.answer(query)

    def test_answer_generation_error_raises(self):
        """
        Synthesizer failures are wrapped as GenerationError at the generation boundary.
        """
        engine = _make_engine()
        query = _make_query()

        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError(
            "generation failed: LLM returned empty"
        )

        with pytest.raises(GenerationError, match="Generation failed"):
            engine.answer(query)

    def test_answer_llm_error_raises_generation_error(self):
        """
        LLM provider failures from the synthesizer are generation failures.
        """
        engine = _make_engine()
        query = _make_query()

        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError("llm api rate limit exceeded")

        with pytest.raises(GenerationError, match="Generation failed"):
            engine.answer(query)

    def test_answer_unknown_synthesis_error_raises_generation_error(self):
        """
        Synthesizer failures do not depend on error-message keywords.
        """
        engine = _make_engine()
        query = _make_query()

        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError("unexpected null pointer")

        with pytest.raises(GenerationError, match="Generation failed"):
            engine.answer(query)

    def test_answer_with_table_results(self):
        """Table handler is invoked after expansion."""
        engine = _make_engine()
        query = _make_query("what is the average salary?")

        engine._retrieval_router.retrieve.return_value = [MagicMock()]

        read_result = MagicMock(name="table_read_result")
        engine._reader.read.return_value = [read_result]

        expanded = [read_result]
        engine._expander.expand.return_value = expanded

        # Override side_effect for this test to verify table_handler is called
        augmented = [MagicMock(name="augmented")]
        engine._table_handler.process.side_effect = None
        engine._table_handler.process.return_value = augmented

        engine._assembler.assemble.return_value = "context"
        engine._synthesizer.generate.return_value = Answer(
            text="The average salary is $50k.",
            provenance=[],
            metadata={},
        )

        engine.answer(query)

        engine._table_handler.process.assert_called_once_with(
            query.text,
            expanded,
            allow_sql_generation=True,
        )
        engine._assembler.assemble.assert_called_once_with(query.text, augmented)


# ---------------------------------------------------------------------------
# TestEvidence
# ---------------------------------------------------------------------------


class TestEvidence:
    """Tests for retrieval-first evidence packs."""

    def test_evidence_skips_chat_prep_synthesis_and_table_sql(self):
        """Evidence mode uses deterministic retrieval prep and does not synthesize."""
        engine = _make_engine()
        engine._query_batcher.batch_classify.side_effect = AssertionError("chat prep called")
        engine._synthesizer.generate.side_effect = AssertionError("synthesis called")

        def _deterministic_table_only(query, results, *, allow_sql_generation=True):
            if allow_sql_generation:
                raise AssertionError("table SQL called")
            return results

        engine._table_handler.process.side_effect = _deterministic_table_only

        address = Address(
            kind=AddressKind.SECTION,
            source_id="doc-1",
            location="Sprint 47",
            summary="Sprint 47 test results",
            score=0.91,
        )
        result = ReadResult(
            address=address,
            content="Sprint 47 failed because the payment retry test timed out.",
            file_path="docs/sprint.md",
            line_range=(10, 12),
        )
        engine._retrieval_router.retrieve.return_value = [address]
        engine._reader.read.return_value = [result]
        engine._expander.expand.return_value = [result]

        decision = MagicMock()
        decision.mode = AnswerMode.SUFFICIENT
        decision.reasons = ("Sources support a confident answer.",)
        engine._governance = MagicMock()
        engine._governance.decide.return_value = decision

        pack = engine.evidence(Query(text="Which test case failed in Sprint 47?"), top_k=1)

        assert pack.mode == AnswerMode.SUFFICIENT
        assert pack.reasons == ["Sources support a confident answer."]
        assert len(pack.items) == 1
        assert pack.items[0].file_path == "docs/sprint.md"
        assert pack.items[0].address_kind == "section"
        engine._query_batcher.batch_classify.assert_not_called()
        engine._synthesizer.generate.assert_not_called()
        engine._table_handler.process.assert_called_once_with(
            "Which test case failed in Sprint 47?",
            [result],
            allow_sql_generation=False,
        )

    def test_evidence_adds_semantic_keywords_to_broad_recall_profile(self):
        """Evidence mode enriches broad recall keywords without full chat prep."""
        engine = _make_engine()
        engine._semantic_keyword_batcher.batch_classify.return_value = BatchResult(
            keywords=["payment retry", "timeout failure"]
        )
        address = Address(
            kind=AddressKind.SECTION,
            source_id="doc-1",
            location="Sprint 47",
            summary="Sprint 47 test results",
            score=0.91,
        )
        result = ReadResult(
            address=address,
            content="Sprint 47 failed because the payment retry test timed out.",
            file_path="docs/sprint.md",
            line_range=(10, 12),
        )
        engine._retrieval_router.retrieve.return_value = [address]
        engine._reader.read.return_value = [result]
        decision = MagicMock()
        decision.mode = AnswerMode.SUFFICIENT
        decision.reasons = ("Enough evidence.",)
        engine._governance = MagicMock()
        engine._governance.decide.return_value = decision

        engine.evidence(Query(text="Which test case failed in Sprint 47?"), top_k=1)

        engine._query_batcher.batch_classify.assert_not_called()
        engine._semantic_keyword_batcher.batch_classify.assert_called_once()
        profile = engine._retrieval_router.retrieve.call_args_list[0].args[1]
        assert "payment retry" in profile.keywords
        assert "timeout failure" in profile.keywords

    def test_evidence_uses_pyrrho_pre_profile_for_comparisons(self):
        """Pre-retrieval profile steering is Pyrrho-owned when query heads exist."""
        engine = _make_engine()
        addresses, results = _evidence_results(2)
        engine._retrieval_router.retrieve.return_value = addresses
        engine._reader.read.return_value = results
        engine._governance = MagicMock()
        engine._governance.decide.return_value = _decision(
            AnswerMode.SUFFICIENT,
            "Enough comparative evidence.",
        )
        engine._governance.plan_query.return_value = MagicMock(
            retrieval_intents=MagicMock(
                final_labels=("needs_lookup", "needs_comparison_or_set"),
                final_label="needs_comparison_or_set",
                confidence=0.9,
                probabilities={"needs_lookup": 0.75, "needs_comparison_or_set": 0.9},
            ),
            evidence_kinds=MagicMock(
                final_labels=("needs_text",),
                final_label="needs_text",
                confidence=0.8,
                probabilities={"needs_text": 0.8},
            ),
        )

        pack = engine.evidence(Query(text="Compare React and Vue performance"), top_k=2)

        profile = engine._retrieval_router.retrieve.call_args_list[0].args[1]
        assert profile.query_contract == "comparison_coverage"
        assert profile.has_comparison_intent is True
        assert profile.answer_type == "comparative"
        assert profile.planning_owner == "pyrrho"
        query_profile = pack.metadata["query_profile"]
        assert "signals" not in query_profile
        assert "pyrrho_pre" in query_profile
        assert query_profile["profile"]["answer_type"] == "comparative"
        assert query_profile["profile"]["planning_owner"] == "pyrrho"

    def test_evidence_uses_pyrrho_cutoff_over_reranked_results(self):
        """Evidence mode returns the smallest reranked prefix Pyrrho accepts."""
        engine = _make_engine()
        addresses, results = _evidence_results(3)
        engine._retrieval_router.retrieve.return_value = addresses
        engine._reader.read.return_value = results

        engine._governance = MagicMock()
        engine._governance.decide.side_effect = [
            _decision(AnswerMode.INSUFFICIENT, "Need more evidence."),
            _decision(AnswerMode.SUFFICIENT, "Enough evidence at two docs."),
        ]

        pack = engine.evidence(Query(text="What happened?"), top_k=3)

        assert pack.mode == AnswerMode.SUFFICIENT
        assert pack.reasons == ["Enough evidence at two docs."]
        assert [item.file_path for item in pack.items] == ["docs/1.md", "docs/2.md"]
        for timing_name in (
            "Query prep",
            "Qwen query keywords",
            "Recall",
            "Rerank",
            "Read",
            "Retrieval",
            "Governance",
        ):
            assert timing_name in pack.timings
        cutoff = pack.metadata["governance_cutoff"]
        assert cutoff["evaluated"] == 2
        assert cutoff["selected"] == 2
        assert cutoff["max"] == 3
        assert cutoff["mode"] == "sufficient"
        assert cutoff["policy"]["query_shape"] == "narrow"
        assert cutoff["pyrrho"] == {
            "mode": "sufficient",
            "probabilities": {
                "insufficient": 0.11,
                "disputed": 0.22,
                "sufficient": 0.67,
            },
            "reason": "Enough evidence at two docs.",
        }
        first_contexts = engine._governance.decide.call_args_list[0].args[1]
        second_contexts = engine._governance.decide.call_args_list[1].args[1]
        assert len(first_contexts) == 1
        assert len(second_contexts) == 2
        engine._expander.expand.assert_not_called()

    def test_broad_query_requires_minimum_sufficient_window(self):
        """Pyrrho broad-answer plans do not stop on a top-1 sufficient verdict."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = addresses
        engine._reader.read.return_value = results
        engine._governance = MagicMock()
        engine._governance.decide.side_effect = [
            _decision(AnswerMode.SUFFICIENT, "Top one is plausible."),
            _decision(AnswerMode.SUFFICIENT, "Top two are plausible."),
            _decision(AnswerMode.SUFFICIENT, "Top three are plausible."),
            _decision(AnswerMode.SUFFICIENT, "Broad window is enough."),
        ]

        pack = engine.evidence(Query(text="Summarize customer feedback themes"), top_k=4)

        assert pack.mode == AnswerMode.SUFFICIENT
        assert [item.file_path for item in pack.items] == [
            "docs/1.md",
            "docs/2.md",
            "docs/3.md",
            "docs/4.md",
        ]
        assert engine._governance.decide.call_count == 4
        cutoff = pack.metadata["governance_cutoff"]
        assert cutoff["policy"]["query_shape"] == "broad"
        assert cutoff["policy"]["min_sufficient_docs"] == 4

    def test_comparison_query_stops_on_disputed_after_two_docs(self):
        """Pyrrho comparison plans can stop on disputed once both sides exist."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = addresses
        engine._reader.read.return_value = results
        engine._governance = MagicMock()
        engine._governance.decide.side_effect = [
            _decision(AnswerMode.INSUFFICIENT, "Need another side."),
            _decision(AnswerMode.DISPUTED, "Sources disagree."),
        ]

        pack = engine.evidence(Query(text="Compare React vs Vue performance"), top_k=4)

        assert pack.mode == AnswerMode.DISPUTED
        assert [item.file_path for item in pack.items] == ["docs/1.md", "docs/2.md"]
        assert engine._governance.decide.call_count == 2
        assert pack.metadata["governance_cutoff"]["policy"]["query_shape"] == "comparison"

    def test_comparative_or_query_requires_both_temporal_sides(self):
        """Pyrrho comparison plans should not stop on one quarterly document."""
        engine = _make_engine()
        addresses, results = _evidence_results(2)
        results[0].content = "Q1 total revenue was 100."
        results[0].address = Address(
            kind=results[0].address.kind,
            source_id=results[0].address.source_id,
            location=results[0].address.location,
            summary="Q1 total revenue was 100.",
            score=results[0].address.score,
            metadata=results[0].address.metadata,
        )
        results[1].content = "Q2 total revenue was 120."
        results[1].address = Address(
            kind=results[1].address.kind,
            source_id=results[1].address.source_id,
            location=results[1].address.location,
            summary="Q2 total revenue was 120.",
            score=results[1].address.score,
            metadata=results[1].address.metadata,
        )
        engine._retrieval_router.retrieve.return_value = addresses
        engine._reader.read.return_value = results
        engine._governance = MagicMock()
        engine._governance.decide.side_effect = [
            _decision(AnswerMode.SUFFICIENT, "Top one is not enough."),
            _decision(AnswerMode.SUFFICIENT, "Both quarters represented."),
        ]

        pack = engine.evidence(
            Query(text="Which quarter had higher total revenue, Q1 or Q2?"),
            top_k=2,
        )

        assert pack.mode == AnswerMode.SUFFICIENT
        assert [item.file_path for item in pack.items] == ["docs/1.md", "docs/2.md"]
        assert engine._governance.decide.call_count == 2
        cutoff = pack.metadata["governance_cutoff"]
        assert cutoff["policy"]["query_shape"] == "comparison"
        assert cutoff["policy"]["min_sufficient_docs"] == 2

    def test_narrow_query_returns_pyrrho_dispute_without_local_patience(self):
        """Narrow queries return Pyrrho's dispute once its evidence floor is met."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = addresses
        engine._reader.read.return_value = results
        engine._governance = MagicMock()
        engine._governance.decide.side_effect = [
            _decision(AnswerMode.INSUFFICIENT, "Need evidence."),
            _decision(AnswerMode.DISPUTED, "Conflict appears.", probs=(0.26, 0.55, 0.19)),
            _decision(AnswerMode.DISPUTED, "Conflict remains.", probs=(0.26, 0.55, 0.19)),
            _decision(AnswerMode.DISPUTED, "Conflict persisted.", probs=(0.26, 0.55, 0.19)),
        ]

        pack = engine.evidence(Query(text="What happened to invoice 17?"), top_k=4)

        assert pack.mode == AnswerMode.DISPUTED
        assert [item.file_path for item in pack.items] == [
            "docs/1.md",
            "docs/2.md",
            "docs/3.md",
            "docs/4.md",
        ]
        assert engine._governance.decide.call_count == 2
        cutoff = pack.metadata["governance_cutoff"]
        assert cutoff["policy"]["query_shape"] == "narrow"
        assert "disputed_patience_docs" not in cutoff["policy"]

    def test_broad_query_disputed_continues_to_cutoff(self):
        """Pyrrho broad-answer plans do not stop on disputed until the cutoff."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = addresses
        engine._reader.read.return_value = results
        engine._governance = MagicMock()
        engine._governance.decide.side_effect = [
            _decision(AnswerMode.DISPUTED, "Conflict one."),
            _decision(AnswerMode.DISPUTED, "Conflict two."),
            _decision(AnswerMode.DISPUTED, "Conflict three."),
            _decision(AnswerMode.DISPUTED, "Conflict four."),
        ]

        pack = engine.evidence(Query(text="Summarize customer feedback themes"), top_k=4)

        assert pack.mode == AnswerMode.DISPUTED
        assert len(pack.items) == 4
        assert engine._governance.decide.call_count == 4
        assert pack.metadata["governance_cutoff"]["policy"]["query_shape"] == "broad"


# ---------------------------------------------------------------------------
# TestPoint
# ---------------------------------------------------------------------------


class TestPoint:
    """Tests for source registration and collection routing."""

    def test_point_collection_override_rebinds_collection_components(self, tmp_path):
        """point(..., collection=...) binds background ingestion to that collection."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(collection="default")
        engine._bg_worker = None
        engine._manifest = None
        engine._source_dir = None
        engine._retrieval_router = MagicMock()
        engine._reader = MagicMock()
        engine._chat_factory = None
        engine._chat = None
        engine._connection_manager = MagicMock()
        engine._table_store = MagicMock()
        engine._sqlite_table_store = MagicMock()
        engine._entity_graph_store = None
        engine._enricher_chat = None
        engine._summarizer_chat = None
        engine._fast_index_code_files = MagicMock()

        source = tmp_path / "docs"
        source.mkdir()
        workspace = tmp_path / ".fitz"
        manifest = MagicMock()

        def _load(collection: str) -> None:
            engine._config.collection = collection

        with (
            patch.object(engine, "load", side_effect=_load) as load,
            patch("fitz_sage.core.paths.FitzPaths.workspace", return_value=workspace),
            patch("fitz_sage.engines.fitz_krag.progressive.builder.ManifestBuilder") as builder_cls,
            patch(
                "fitz_sage.engines.fitz_krag.retrieval.strategies.agentic_search"
                ".AgenticSearchStrategy"
            ),
            patch(
                "fitz_sage.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline"
            ) as pipeline_cls,
        ):
            builder_cls.return_value.build.return_value = manifest

            engine.point(source, "custom", start_worker=False)

        load.assert_called_once_with("custom")
        pipeline_cls.assert_called_once()
        assert pipeline_cls.call_args.kwargs["collection"] == "custom"
        build_manifest_path = builder_cls.return_value.build.call_args.args[1]
        assert build_manifest_path == workspace / "collections" / "custom" / "manifest.json"

    def test_point_prepares_managed_qwen_snapshot_for_sources(self, tmp_path):
        """Source registration validates managed Qwen model files early."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(collection="default")
        engine._bg_worker = None
        engine._manifest = None
        engine._source_dir = None
        engine._retrieval_router = MagicMock()
        engine._reader = MagicMock()
        engine._chat_factory = None
        engine._chat = None
        engine._connection_manager = MagicMock()
        engine._table_store = MagicMock()
        engine._sqlite_table_store = MagicMock()
        engine._entity_graph_store = None
        engine._summarizer_chat = MagicMock()
        engine._fast_index_code_files = MagicMock()

        model_info = MagicMock()
        model_info.revision = "abc123456789def"
        engine._enricher_chat = MagicMock()
        engine._enricher_chat.ensure_available.return_value = model_info

        source = tmp_path / "docs"
        source.mkdir()
        workspace = tmp_path / ".fitz"
        manifest = MagicMock()
        manifest.entries.return_value = {"release_notes.md": MagicMock()}
        progress = MagicMock()

        with (
            patch("fitz_sage.core.paths.FitzPaths.workspace", return_value=workspace),
            patch("fitz_sage.engines.fitz_krag.progressive.builder.ManifestBuilder") as builder_cls,
            patch(
                "fitz_sage.engines.fitz_krag.retrieval.strategies.agentic_search"
                ".AgenticSearchStrategy"
            ),
            patch("fitz_sage.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline"),
        ):
            builder_cls.return_value.build.return_value = manifest

            engine.point(source, start_worker=False, progress=progress)

        engine._enricher_chat.ensure_available.assert_called_once()
        progress.assert_any_call("Preparing managed Qwen3 0.6B ONNX GenAI enrichment snapshot...")
        progress.assert_any_call("Managed Qwen snapshot ready (abc123456789).")

    def test_point_deletes_stale_files_outside_current_manifest(self, tmp_path):
        """Re-pointing cleans previously indexed files that the scanner now excludes."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(collection="rag_test_corpus")
        engine._bg_worker = None
        engine._manifest = None
        engine._source_dir = None
        engine._retrieval_router = MagicMock()
        engine._reader = MagicMock()
        engine._chat_factory = None
        engine._chat = None
        engine._connection_manager = MagicMock()
        engine._table_store = MagicMock()
        engine._sqlite_table_store = MagicMock()
        engine._entity_graph_store = None
        engine._enricher_chat = None
        engine._summarizer_chat = None
        engine._fast_index_code_files = MagicMock()

        source = tmp_path / "rag_test_corpus"
        source.mkdir()
        workspace = source / ".fitz"
        manifest = MagicMock()
        manifest.entries.return_value = {
            "README.md": MagicMock(),
            "hierarchical_rag/feedback.md": MagicMock(),
        }
        pipeline_core = MagicMock()

        with (
            patch("fitz_sage.core.paths.FitzPaths.workspace", return_value=workspace),
            patch("fitz_sage.engines.fitz_krag.progressive.builder.ManifestBuilder") as builder_cls,
            patch(
                "fitz_sage.engines.fitz_krag.retrieval.strategies.agentic_search"
                ".AgenticSearchStrategy"
            ),
            patch(
                "fitz_sage.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline",
                return_value=pipeline_core,
            ),
        ):
            builder_cls.return_value.build.return_value = manifest

            engine.point(source, start_worker=False)

        pipeline_core.delete_files_not_in_paths.assert_called_once_with(
            {"README.md", "hierarchical_rag/feedback.md"}
        )

    def test_continue_indexing_runs_worker_to_deep_completion(self, tmp_path):
        """Persisted indexing resumes without rebuilding the manifest."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._manifest = MagicMock()
        engine._source_dir = tmp_path / "docs"
        engine._build_ingest_core = MagicMock(return_value=MagicMock())

        with patch(
            "fitz_sage.engines.fitz_krag.progressive.worker.BackgroundIngestWorker"
        ) as worker_cls:
            worker = worker_cls.return_value

            engine.continue_indexing()

        worker_cls.assert_called_once()
        assert worker_cls.call_args.kwargs["manifest"] is engine._manifest
        assert worker_cls.call_args.kwargs["source_dir"] == engine._source_dir
        worker.run_until_deep_complete.assert_called_once()


# ---------------------------------------------------------------------------
# TestConfig
# ---------------------------------------------------------------------------


class TestConfig:
    """Tests for the config property."""

    def test_config_property_returns_config(self):
        """The config property returns the stored FitzKragConfig."""
        engine = _make_engine(collection="my_project")
        assert isinstance(engine.config, FitzKragConfig)
        assert engine.config.collection == "my_project"

    def test_config_property_reflects_overrides(self):
        """Config overrides are reflected in the property."""
        engine = _make_engine(
            collection="custom",
            top_read=10,
            top_addresses=20,
        )
        assert engine.config.top_read == 10
        assert engine.config.top_addresses == 20
        assert engine.config.collection == "custom"
