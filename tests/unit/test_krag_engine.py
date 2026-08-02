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
    QueryIntelligenceError,
)
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine, _build_provider_config
from fitz_sage.engines.fitz_krag.progressive.write_lock import (
    CollectionBusyError,
    CollectionWriteLock,
)
from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouterResponse
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


def _make_query(text: str = "How does auth work?") -> Query:
    """Return a real query so metadata defaults match the public contract."""
    return Query(text=text)


def _decision(
    mode: AnswerMode,
    reason: str,
    *,
    probs: tuple[float, float, float] | None = None,
) -> MagicMock:
    """Build a lightweight authoritative Pyrrho decision."""
    decision = MagicMock()
    decision.verdict = mode.value.upper()
    decision.reasons = (reason,)
    decision.reason = reason
    if probs is not None:
        probabilities = probs
    elif mode is AnswerMode.SUFFICIENT:
        probabilities = (0.11, 0.22, 0.67)
    elif mode is AnswerMode.DISPUTED:
        probabilities = (0.12, 0.68, 0.20)
    else:
        probabilities = (0.69, 0.18, 0.13)
    decision.to_dict.return_value = {
        "schema_version": 1,
        "verdict": decision.verdict,
        "reason": reason,
        "probabilities": {
            "INSUFFICIENT": probabilities[0],
            "DISPUTED": probabilities[1],
            "SUFFICIENT": probabilities[2],
        },
    }
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

    def test_load_same_collection_does_not_reinitialize_components(self, _patches):
        """The common create-then-load lifecycle initializes components once."""
        config = _make_config(collection="default")
        engine = FitzKragEngine(config)

        with patch.object(engine, "_try_load_persisted_manifest"):
            engine.load("default")

        _patches["ensure_schema"].assert_called_once()
        _patches["RetrievalRouter"].assert_called_once()

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
        """Synthesis consumes the same governed evidence set as evidence()."""
        engine = _make_engine()
        query = _make_query(
            "What does the login function do when the user provides invalid credentials?"
        )
        addresses, results = _evidence_results(2)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        context = "governed context"
        engine._assembler.assemble.return_value = context

        expected_answer = Answer(
            text="The login function authenticates users.",
            provenance=[Provenance(source_id="auth.py:42")],
            metadata={"engine": "fitz_krag"},
        )
        engine._synthesizer.generate.return_value = expected_answer

        result = engine.answer(query)

        assert engine._retrieval_router.retrieve.call_count >= 1
        call_args = engine._retrieval_router.retrieve.call_args_list[0]
        assert call_args[0][0] == query.text
        from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile

        assert isinstance(call_args[0][1], RetrievalProfile)
        assert call_args[1]["rewrite_result"] is None
        assert engine._reader.read.call_args_list[0] == call(
            addresses,
            engine._config.top_read,
        )
        assert engine._expander.expand.call_count >= 1
        assert engine._table_handler.process.call_args_list[0] == call(
            query.text,
            results,
            allow_sql_generation=True,
        )
        selected = engine._synthesizer.generate.call_args.args[2]
        engine._assembler.assemble.assert_called_once_with(query.text, selected)

        engine._synthesizer.generate.assert_called_once_with(
            query.text,
            context,
            selected,
            answer_mode=AnswerMode.SUFFICIENT,
            gap_context=None,
            conflict_context=None,
        )

        assert result is expected_answer
        assert result.metadata["pyrrho"]["verdict"] == "SUFFICIENT"
        assert result.metadata["query_profile"]["profile"]["top_k"] == engine._config.top_addresses

    def test_answer_uses_no_chat_query_planner_by_default(self):
        """Default query prep is deterministic unless query_intelligence is configured."""
        engine = _make_engine()
        query = _make_query("Compare Q1 2024 vs Q2 2024 API failures")
        engine._query_batcher.batch_classify.side_effect = AssertionError("chat prep called")

        addresses, results = _evidence_results(2)
        results[0].content = "Q1 2024 API failures totaled 8."
        results[1].content = "Q2 2024 API failures totaled 5."
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._assembler.assemble.return_value = MagicMock()
        expected = Answer(text="Answer.", provenance=[], metadata={})
        engine._synthesizer.generate.return_value = expected

        result = engine.answer(query)

        assert result is expected
        engine._query_batcher.batch_classify.assert_not_called()
        call_args = engine._retrieval_router.retrieve.call_args_list[0]
        profile = call_args[0][1]
        assert "q1 2024" in profile.temporal_references
        assert "q2 2024" in profile.temporal_references
        assert profile.planning_owner == "hybrid"
        engine._pyrrho.plan_query.assert_called_once_with(query.text)

    def test_answer_empty_query_raises(self):
        """Empty or whitespace-only query text raises QueryError."""
        engine = _make_engine()

        for blank in ("", "   ", "\t\n"):
            q = MagicMock(text=blank, metadata={})
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

        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse([])

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

        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse([MagicMock()])
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

        addresses, results = _evidence_results(1)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
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

        addresses, results = _evidence_results(1)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
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

        addresses, results = _evidence_results(1)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError("unexpected null pointer")

        with pytest.raises(GenerationError, match="Generation failed"):
            engine.answer(query)

    def test_answer_with_table_results(self):
        """Table synthesis uses deterministic table evidence from the governed path."""
        engine = _make_engine()
        query = _make_query("what is the average salary?")
        address = Address(
            kind=AddressKind.TABLE,
            source_id="table-1",
            location="salaries",
            summary="Salary table",
            score=0.9,
        )
        read_result = ReadResult(
            address=address,
            content="| employee | salary |\n| A | 50000 |",
            file_path="salaries.csv",
        )
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse([address])
        engine._reader.read.return_value = [read_result]
        augmented = [read_result]
        engine._table_handler.process.side_effect = None
        engine._table_handler.process.return_value = augmented

        engine._assembler.assemble.return_value = "context"
        engine._synthesizer.generate.return_value = Answer(
            text="The average salary is $50k.",
            provenance=[],
            metadata={},
        )

        engine.answer(query)

        assert engine._table_handler.process.call_args_list[0] == call(
            query.text,
            [read_result],
            allow_sql_generation=True,
        )
        assert all(
            invocation.kwargs["allow_sql_generation"] is True
            for invocation in engine._table_handler.process.call_args_list
        )
        assembled_results = engine._assembler.assemble.call_args.args[1]
        synthesized_results = engine._synthesizer.generate.call_args.args[2]
        assert assembled_results == synthesized_results
        assert [result.content for result in assembled_results] == [read_result.content]


# ---------------------------------------------------------------------------
# TestEvidence
# ---------------------------------------------------------------------------


class TestEvidence:
    """Tests for retrieval-first evidence packs."""

    def test_evidence_skips_chat_prep_and_synthesis_but_allows_table_queries(self):
        """Evidence uses deterministic prep, runs table retrieval, and does not synthesize."""
        engine = _make_engine()
        engine._query_batcher.batch_classify.side_effect = AssertionError("chat prep called")
        engine._synthesizer.generate.side_effect = AssertionError("synthesis called")

        def _table_query(query, results, *, allow_sql_generation=True):
            assert allow_sql_generation is True
            return results

        engine._table_handler.process.side_effect = _table_query

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
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse([address])
        engine._reader.read.return_value = [result]
        engine._expander.expand.return_value = [result]

        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.SUFFICIENT,
            "Sources support a confident answer.",
        )

        pack = engine.evidence(Query(text="Which test case failed in Sprint 47?"), top_k=1)

        assert pack.mode == AnswerMode.SUFFICIENT
        assert pack.reasons == ["Sources support a confident answer."]
        assert len(pack.items) == 1
        assert pack.items[0].file_path == "docs/sprint.md"
        assert pack.items[0].address_kind == "section"
        engine._query_batcher.batch_classify.assert_not_called()
        engine._synthesizer.generate.assert_not_called()
        assert engine._table_handler.process.call_args_list[0] == call(
            "Which test case failed in Sprint 47?",
            [result],
            allow_sql_generation=True,
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
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse([address])
        engine._reader.read.return_value = [result]
        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.SUFFICIENT,
            "Enough evidence.",
        )

        pack = engine.evidence(Query(text="Which test case failed in Sprint 47?"), top_k=1)

        engine._query_batcher.batch_classify.assert_not_called()
        engine._semantic_keyword_batcher.batch_classify.assert_called_once()
        profile = engine._retrieval_router.retrieve.call_args_list[0].args[1]
        assert "payment retry" in profile.keywords
        assert "timeout failure" in profile.keywords
        assert pack.metadata["retrieval_trace"]["semantic_query_expansion"] == {
            "enabled": True,
            "used": True,
            "status": "expanded",
            "added_keywords": 2,
        }

    def test_evidence_falls_back_when_semantic_keyword_expansion_is_malformed(self):
        """Optional semantic expansion cannot make literal retrieval unavailable."""
        engine = _make_engine()
        engine._semantic_keyword_batcher.batch_classify.side_effect = QueryIntelligenceError(
            "batched query intelligence missing `keywords` array"
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
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse([address])
        engine._reader.read.return_value = [result]

        pack = engine.evidence(Query(text="Which test case failed in Sprint 47?"), top_k=1)

        assert [item.file_path for item in pack.items] == ["docs/sprint.md"]
        engine._retrieval_router.retrieve.assert_called_once()
        profile = engine._retrieval_router.retrieve.call_args.args[1]
        assert "payment retry" not in profile.keywords
        assert pack.metadata["retrieval_trace"]["semantic_query_expansion"] == {
            "enabled": True,
            "used": False,
            "status": "failed",
            "added_keywords": 0,
            "error_type": "QueryIntelligenceError",
            "error": "batched query intelligence missing `keywords` array",
        }

    def test_evidence_uses_pyrrho_pre_profile_for_comparisons(self):
        """Pre-retrieval profile steering is Pyrrho-owned when query heads exist."""
        engine = _make_engine()
        addresses, results = _evidence_results(2)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.SUFFICIENT,
            "Enough comparative evidence.",
        )
        engine._pyrrho.plan_query.return_value = MagicMock(
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
        assert profile.planning_owner == "hybrid"
        query_profile = pack.metadata["query_profile"]
        assert "signals" not in query_profile
        assert "pyrrho_pre" in query_profile
        assert query_profile["profile"]["answer_type"] == "comparative"
        assert query_profile["profile"]["planning_owner"] == "hybrid"

    def test_evidence_uses_fixed_budget_and_one_authoritative_pyrrho_call(self):
        """Evidence delivery is fixed before Pyrrho and does not depend on its verdict."""
        engine = _make_engine()
        addresses, results = _evidence_results(3)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results

        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.SUFFICIENT,
            "Enough evidence.",
        )

        pack = engine.evidence(Query(text="What happened?"), top_k=2)

        assert pack.mode == AnswerMode.SUFFICIENT
        assert pack.reasons == ["Enough evidence."]
        assert [item.file_path for item in pack.items] == ["docs/1.md", "docs/2.md"]
        for timing_name in (
            "Query prep",
            "Qwen query keywords",
            "Recall",
            "Rerank",
            "Read",
            "Retrieval",
            "Pyrrho",
        ):
            assert timing_name in pack.timings
        assert pack.metadata["evidence_delivery"] == {
            "available": 3,
            "selected": 2,
            "limit": 2,
        }
        assert pack.metadata["pyrrho"] == engine._pyrrho.decide.return_value.to_dict()
        engine._pyrrho.decide.assert_called_once()
        contexts = engine._pyrrho.decide.call_args.args[1]
        assert contexts == [
            {"source_id": "doc-1", "text": results[0].content},
            {"source_id": "doc-2", "text": results[1].content},
        ]
        engine._expander.expand.assert_called_once_with(
            results,
            entity_expansion_limit=3,
        )

    def test_broad_query_uses_requested_budget_without_local_evidence_floor(self):
        """Broad query shape cannot delay or override Pyrrho's verdict."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.SUFFICIENT,
            "Pyrrho accepts the evidence.",
        )

        pack = engine.evidence(Query(text="Summarize customer feedback themes"), top_k=1)

        assert pack.mode == AnswerMode.SUFFICIENT
        assert [item.file_path for item in pack.items] == ["docs/1.md"]
        engine._pyrrho.decide.assert_called_once()

    def test_comparison_query_returns_exact_pyrrho_dispute(self):
        """Comparison query shape does not alter Pyrrho's disputed verdict."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.DISPUTED,
            "Sources disagree.",
        )

        pack = engine.evidence(Query(text="Compare React vs Vue performance"), top_k=4)

        assert pack.mode == AnswerMode.DISPUTED
        assert len(pack.items) == 4
        engine._pyrrho.decide.assert_called_once()

    def test_comparative_query_delivers_both_requested_temporal_sides(self):
        """Retrieval shape provides both sides before the single Pyrrho call."""
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
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.SUFFICIENT,
            "Both quarters represented.",
        )

        pack = engine.evidence(
            Query(text="Which quarter had higher total revenue, Q1 or Q2?"),
            top_k=2,
        )

        assert pack.mode == AnswerMode.SUFFICIENT
        assert [item.file_path for item in pack.items] == ["docs/1.md", "docs/2.md"]
        engine._pyrrho.decide.assert_called_once()

    def test_narrow_query_returns_pyrrho_dispute_without_local_policy(self):
        """Narrow queries return Pyrrho's disputed verdict unchanged."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.DISPUTED,
            "Conflict appears.",
            probs=(0.26, 0.55, 0.19),
        )

        pack = engine.evidence(Query(text="What happened to invoice 17?"), top_k=4)

        assert pack.mode == AnswerMode.DISPUTED
        assert [item.file_path for item in pack.items] == [
            "docs/1.md",
            "docs/2.md",
            "docs/3.md",
            "docs/4.md",
        ]
        engine._pyrrho.decide.assert_called_once()

    def test_broad_query_returns_pyrrho_dispute_without_local_override(self):
        """Broad queries also return Pyrrho's disputed verdict unchanged."""
        engine = _make_engine()
        addresses, results = _evidence_results(4)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._pyrrho = MagicMock()
        engine._pyrrho.decide.return_value = _decision(
            AnswerMode.DISPUTED,
            "Conflict.",
        )

        pack = engine.evidence(Query(text="Summarize customer feedback themes"), top_k=4)

        assert pack.mode == AnswerMode.DISPUTED
        assert len(pack.items) == 4
        engine._pyrrho.decide.assert_called_once()

    def test_trace_records_the_canonical_governed_execution(self, monkeypatch, tmp_path):
        """A trace captures compiled and Pyrrho-delivered evidence without rerunning."""
        from fitz_sage.core.paths import FitzPaths

        engine = _make_engine()
        addresses, results = _evidence_results(3)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._semantic_keyword_batcher.batch_classify.return_value = BatchResult(
            keywords=["incident"]
        )
        monkeypatch.setattr(FitzPaths, "workspace", classmethod(lambda cls: tmp_path))

        run = engine.trace(Query(text="What happened?"), top_k=3)

        assert engine._retrieval_router.retrieve.call_count == 1
        assert run.evidence.mode == AnswerMode.SUFFICIENT
        assert len(run.ranked_evidence) == 3
        assert run.ranked_evidence[0].content == results[0].content
        assert run.pyrrho.evidence_count == len(run.evidence.items)
        assert len(run.pyrrho_evidence) == len(run.evidence.items)
        assert run.pyrrho.decision == run.evidence.metadata["pyrrho"]
        assert run.environment.engine == "fitz_krag"
        assert run.environment.collection == "test_collection"
        assert any(term.origin == "literal" for term in run.query.terms)
        assert any(
            term.text == "incident" and term.origin == "semantic" for term in run.query.terms
        )

    def test_trace_labels_query_intelligence_keyword_fallback_as_deterministic(
        self, monkeypatch, tmp_path
    ):
        """Configured query intelligence does not claim fallback term ownership."""
        from fitz_sage.core.paths import FitzPaths

        engine = _make_engine(query_intelligence="endpoint/qwen2.5-7b-instruct")
        addresses, results = _evidence_results(1)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        monkeypatch.setattr(
            FitzPaths,
            "workspace",
            classmethod(lambda cls: tmp_path),
        )

        run = engine.trace(Query(text="What happened?"), top_k=1)

        assert any(
            term.text == "happened" and term.origin == "deterministic" for term in run.query.terms
        )
        assert not any(term.origin == "query_intelligence" for term in run.query.terms)

    def test_trace_labels_keywords_returned_by_query_intelligence(self, monkeypatch, tmp_path):
        """Model-produced query terms retain their actual producer."""
        from fitz_sage.core.paths import FitzPaths

        engine = _make_engine(query_intelligence="endpoint/qwen2.5-7b-instruct")
        addresses, results = _evidence_results(1)
        engine._retrieval_router.retrieve.return_value = RetrievalRouterResponse(addresses)
        engine._reader.read.return_value = results
        engine._query_batcher.batch_classify.side_effect = None
        engine._query_batcher.batch_classify.return_value = BatchResult(keywords=["diagnostic"])
        monkeypatch.setattr(
            FitzPaths,
            "workspace",
            classmethod(lambda cls: tmp_path),
        )

        run = engine.trace(Query(text="What happened?"), top_k=1)

        assert any(
            term.text == "diagnostic" and term.origin == "query_intelligence"
            for term in run.query.terms
        )


# ---------------------------------------------------------------------------
# TestPoint
# ---------------------------------------------------------------------------


class TestPoint:
    """Tests for synchronous source indexing and collection routing."""

    def test_point_rejects_a_concurrent_collection_writer(self, tmp_path):
        """Discovery cannot begin while another writer owns the collection."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(collection="default")
        engine._enrichment_worker = None
        source = tmp_path / "docs"
        source.mkdir()
        workspace = tmp_path / ".fitz"
        lock = CollectionWriteLock(
            workspace / "collections" / "default",
            collection="default",
            operation="background enrichment",
        )

        with (
            lock,
            patch("fitz_sage.core.paths.FitzPaths.workspace", return_value=workspace),
            patch("fitz_sage.engines.fitz_krag.progressive.builder.ManifestBuilder") as builder_cls,
            pytest.raises(CollectionBusyError, match="Collection 'default' is busy"),
        ):
            engine.point(source, start_worker=False)

        builder_cls.assert_not_called()

    def test_point_collection_override_rebinds_collection_components(self, tmp_path):
        """point(..., collection=...) binds ingestion to that collection."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(collection="default")
        engine._enrichment_worker = None
        engine._manifest = None
        engine._source_dir = None
        engine._chat_factory = None
        engine._chat = None
        engine._connection_manager = MagicMock()
        engine._table_store = MagicMock()
        engine._sqlite_table_store = MagicMock()
        engine._entity_graph_store = None
        engine._enricher_chat = None
        engine._summarizer_chat = None

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

    def test_point_indexes_registered_files_without_loading_qwen(self, tmp_path):
        """Every supported changed file is searchable before point returns."""
        from fitz_sage.engines.fitz_krag.progressive.manifest import FileState

        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(collection="default")
        engine._enrichment_worker = None
        engine._manifest = None
        engine._source_dir = None
        engine._chat_factory = None
        engine._chat = None
        engine._connection_manager = MagicMock()
        engine._table_store = MagicMock()
        engine._sqlite_table_store = MagicMock()
        engine._entity_graph_store = None
        engine._enricher_chat = MagicMock()
        engine._summarizer_chat = MagicMock()

        source = tmp_path / "docs"
        source.mkdir()
        document = source / "release_notes.md"
        document.write_text("# Release\nSearchable content", encoding="utf-8")
        workspace = tmp_path / ".fitz"
        manifest = MagicMock()
        entry = MagicMock(
            rel_path="release_notes.md",
            abs_path=str(document),
            file_id="file-1",
            state=FileState.REGISTERED,
        )
        manifest.entries.return_value = {"release_notes.md": entry}
        progress = MagicMock()
        pipeline = MagicMock()

        with (
            patch("fitz_sage.core.paths.FitzPaths.workspace", return_value=workspace),
            patch("fitz_sage.engines.fitz_krag.progressive.builder.ManifestBuilder") as builder_cls,
            patch(
                "fitz_sage.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline",
                return_value=pipeline,
            ),
        ):
            builder_cls.return_value.build.return_value = manifest

            engine.point(source, start_worker=False, progress=progress)

        pipeline.parse_file.assert_called_once_with(
            "release_notes.md",
            document,
            "file-1",
        )
        pipeline.resolve_imports.assert_called_once()
        manifest.update_state.assert_called_once_with("release_notes.md", FileState.INDEXED)
        engine._enricher_chat.ensure_available.assert_not_called()
        progress.assert_any_call("Searchable source index ready (1/1 changed files).")

    def test_point_deletes_stale_files_outside_current_manifest(self, tmp_path):
        """Re-pointing cleans previously indexed files that the scanner now excludes."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._config = _make_config(collection="rag_test_corpus")
        engine._enrichment_worker = None
        engine._manifest = None
        engine._source_dir = None
        engine._chat_factory = None
        engine._chat = None
        engine._connection_manager = MagicMock()
        engine._table_store = MagicMock()
        engine._sqlite_table_store = MagicMock()
        engine._entity_graph_store = None
        engine._enricher_chat = None
        engine._summarizer_chat = None

        source = tmp_path / "rag_test_corpus"
        source.mkdir()
        workspace = source / ".fitz"
        manifest = MagicMock()
        manifest.entries.return_value = {}
        pipeline_core = MagicMock()

        with (
            patch("fitz_sage.core.paths.FitzPaths.workspace", return_value=workspace),
            patch("fitz_sage.engines.fitz_krag.progressive.builder.ManifestBuilder") as builder_cls,
            patch(
                "fitz_sage.engines.fitz_krag.ingestion.pipeline.KragIngestPipeline",
                return_value=pipeline_core,
            ),
        ):
            builder_cls.return_value.build.return_value = manifest

            engine.point(source, start_worker=False)

        pipeline_core.delete_files_not_in_paths.assert_called_once_with(set())

    def test_continue_enrichment_runs_worker_to_completion(self, tmp_path):
        """Persisted enrichment resumes without rebuilding the source index."""
        engine = FitzKragEngine.__new__(FitzKragEngine)
        engine._manifest = MagicMock()
        engine._manifest.path = tmp_path / "manifest.json"
        engine._source_dir = tmp_path / "docs"
        engine._config = _make_config(collection="default")
        engine._build_ingest_core = MagicMock(return_value=MagicMock())

        with patch(
            "fitz_sage.engines.fitz_krag.progressive.worker.BackgroundEnrichmentWorker"
        ) as worker_cls:
            worker = worker_cls.return_value

            engine.continue_enrichment()

        worker_cls.assert_called_once()
        assert worker_cls.call_args.kwargs["manifest"] is engine._manifest
        assert worker_cls.call_args.kwargs["write_lock"].path == tmp_path / "writer.lock"
        engine._manifest.load.assert_called_once()
        engine._manifest.prepare_enrichment_retry.assert_called_once()
        worker.run_until_complete.assert_called_once()


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
