# tests/unit/test_krag_engine.py
"""
Unit tests for FitzKragEngine.

All dependencies are mocked. The engine is constructed via __new__ with
mocked internals for answer/ingest/config tests. Only the init tests
exercise the real __init__ with patched imports.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from fitz_sage.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Provenance,
    QueryError,
)
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

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
    engine._table_handler = MagicMock(name="table_handler")
    engine._table_handler.process.side_effect = lambda q, results: results
    engine._assembler = MagicMock(name="assembler")
    engine._synthesizer = MagicMock(name="synthesizer")
    engine._governance = None
    engine._detection_orchestrator = None
    engine._table_store = MagicMock(name="table_store")
    engine._pg_table_store = MagicMock(name="pg_table_store")
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
    # Configure batcher to return sensible defaults so batched dispatch works
    from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
    from fitz_sage.engines.fitz_krag.query_batcher import BatchResult

    def _default_batch_classify(query, **kwargs):
        return BatchResult(
            analysis=QueryAnalysis(
                primary_type=QueryType.GENERAL, confidence=0.8, refined_query=query
            ),
            detection_results=None,
            rewrite_result=None,
        )

    engine._query_batcher = MagicMock(name="query_batcher")
    engine._query_batcher.batch_classify.side_effect = _default_batch_classify
    return engine


def _make_query(text: str = "How does auth work?") -> MagicMock:
    """Return a mock Query with the given text."""
    q = MagicMock(name="query")
    q.text = text
    return q


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
            "RawFileStore": (
                "fitz_sage.engines.fitz_krag.ingestion" ".raw_file_store.RawFileStore"
            ),
            "SymbolStore": ("fitz_sage.engines.fitz_krag.ingestion" ".symbol_store.SymbolStore"),
            "ImportGraphStore": (
                "fitz_sage.engines.fitz_krag.ingestion" ".import_graph_store.ImportGraphStore"
            ),
            "SectionStore": ("fitz_sage.engines.fitz_krag.ingestion" ".section_store.SectionStore"),
            "TableStoreKrag": ("fitz_sage.engines.fitz_krag.ingestion" ".table_store.TableStore"),
            "ensure_schema": ("fitz_sage.engines.fitz_krag.ingestion" ".schema.ensure_schema"),
            # strategies
            "CodeSearchStrategy": (
                "fitz_sage.engines.fitz_krag.retrieval" ".strategies.code_search.CodeSearchStrategy"
            ),
            "SectionSearchStrategy": (
                "fitz_sage.engines.fitz_krag.retrieval"
                ".strategies.section_search.SectionSearchStrategy"
            ),
            "TableSearchStrategy": (
                "fitz_sage.engines.fitz_krag.retrieval"
                ".strategies.table_search.TableSearchStrategy"
            ),
            # retrieval
            "RetrievalRouter": ("fitz_sage.engines.fitz_krag.retrieval" ".router.RetrievalRouter"),
            "ContentReader": ("fitz_sage.engines.fitz_krag.retrieval" ".reader.ContentReader"),
            "CodeExpander": ("fitz_sage.engines.fitz_krag.retrieval" ".expander.CodeExpander"),
            "TableQueryHandler": (
                "fitz_sage.engines.fitz_krag.retrieval" ".table_handler.TableQueryHandler"
            ),
            # query analysis
            "QueryAnalyzer": ("fitz_sage.engines.fitz_krag" ".query_analyzer.QueryAnalyzer"),
            # context + generation
            "ContextAssembler": (
                "fitz_sage.engines.fitz_krag.context" ".assembler.ContextAssembler"
            ),
            "CodeSynthesizer": (
                "fitz_sage.engines.fitz_krag.generation" ".synthesizer.CodeSynthesizer"
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

        # Core clients called
        _patches["get_chat"].assert_called_once()
        _patches["SqliteConnectionManager"].get_instance.assert_called()

        # Schema ensured
        _patches["ensure_schema"].assert_called_once()

        # Strategies and router created
        _patches["CodeSearchStrategy"].assert_called_once()
        _patches["SectionSearchStrategy"].assert_called_once()
        _patches["TableSearchStrategy"].assert_called_once()
        _patches["RetrievalRouter"].assert_called_once()

        # Reader, expander, analyzer, assembler, synthesizer, table handler created
        _patches["ContentReader"].assert_called_once()
        _patches["CodeExpander"].assert_called_once()
        _patches["TableQueryHandler"].assert_called_once()
        _patches["QueryAnalyzer"].assert_called_once()
        _patches["ContextAssembler"].assert_called_once()
        _patches["CodeSynthesizer"].assert_called_once()

        # Config stored correctly
        assert engine.config is config

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
        # _query_analyzer is not called in the batched dispatch path
        engine._query_analyzer.analyze.assert_not_called()
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
        engine._table_handler.process.assert_called_once_with(query.text, expanded)
        engine._assembler.assemble.assert_called_once_with(
            query.text,
            expanded,
        )
        from fitz_sage.core.answer_mode import AnswerMode

        engine._synthesizer.generate.assert_called_once_with(
            query.text,
            context,
            expanded,
            answer_mode=AnswerMode.TRUSTWORTHY,
            gap_context=None,
            conflict_context=None,
        )

        assert result is expected_answer

    def test_answer_empty_query_raises(self):
        """Empty or whitespace-only query text raises QueryError."""
        engine = _make_engine()

        for blank in ("", "   ", "\t\n"):
            q = _make_query(blank)
            with pytest.raises(QueryError, match="empty"):
                engine.answer(q)

    def test_answer_no_addresses_returns_fallback(self):
        """Router returning [] yields an actionable ABSTAIN answer."""
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = []

        result = engine.answer(query)

        assert result.provenance == []
        assert result.mode == AnswerMode.ABSTAIN
        assert result.metadata["engine"] == "fitz_krag"
        assert result.metadata["query"] == query.text
        assert result.metadata["answer_mode"] == "abstain"
        assert "gap_context" in result.metadata

        # Reader should never be called
        engine._reader.read.assert_not_called()

    def test_answer_no_read_results_returns_fallback(self):
        """Reader returning [] yields a symbols-found fallback Answer."""
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = []

        result = engine.answer(query)

        assert "Found matching symbols" in result.text
        assert result.provenance == []
        assert result.metadata["engine"] == "fitz_krag"

        # Expander should never be called
        engine._expander.expand.assert_not_called()

    def test_answer_retrieval_error_raises_knowledge_error(self):
        """
        When the retrieval router raises an error whose message
        contains 'search', it is wrapped as KnowledgeError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
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

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.side_effect = RuntimeError("retrieval timeout")

        with pytest.raises(KnowledgeError, match="Retrieval failed"):
            engine.answer(query)

    def test_answer_generation_error_raises(self):
        """
        When the synthesizer raises an error whose message contains
        'generation', it is wrapped as GenerationError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
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
        Error message containing 'llm' maps to GenerationError.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError("llm api rate limit exceeded")

        with pytest.raises(GenerationError, match="Generation failed"):
            engine.answer(query)

    def test_answer_unknown_error_raises_knowledge_error(self):
        """
        Errors that don't match 'retriev', 'search', 'generat',
        or 'llm' are wrapped as KnowledgeError with 'KRAG pipeline
        error' message.
        """
        engine = _make_engine()
        query = _make_query()

        engine._query_analyzer.analyze.return_value = MagicMock()
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.side_effect = RuntimeError("unexpected null pointer")

        with pytest.raises(KnowledgeError, match="KRAG pipeline error"):
            engine.answer(query)

    def test_answer_with_table_results(self):
        """Table handler is invoked after expansion."""
        engine = _make_engine()
        query = _make_query("what is the average salary?")

        engine._query_analyzer.analyze.return_value = MagicMock()
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

        engine._table_handler.process.assert_called_once_with(query.text, expanded)
        engine._assembler.assemble.assert_called_once_with(query.text, augmented)


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
