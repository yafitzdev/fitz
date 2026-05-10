# tests/unit/test_krag_detection.py
"""
Unit tests for shared detection integration in the Fitz KRAG engine.

Covers:
- Detection orchestrator initialization (enabled/disabled)
- Router.retrieve behavior with and without detection
- Detection summary flowing through engine.answer() to the router
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

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
    Sets _detection_orchestrator, _cloud_client, _constraints, and _governor
    to None by default (detection/cloud/guardrails disabled).
    """
    config = _make_config(**config_overrides)
    engine = FitzKragEngine.__new__(FitzKragEngine)
    engine._config = config
    engine._chat = MagicMock(name="chat")
    engine._embedder = MagicMock(name="embedder")
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
    engine._detection_orchestrator = None
    engine._cloud_client = None
    engine._constraints = []
    engine._governor = None
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
    engine._hyde_generator = None

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


def _make_address(
    source_id: str = "src1",
    location: str = "mod.func",
    score: float = 0.9,
) -> Address:
    return Address(
        kind=AddressKind.SYMBOL,
        source_id=source_id,
        location=location,
        summary="test",
        score=score,
    )


def _make_detection_summary(
    query_variations: list[str] | None = None,
    comparison_entities: list[str] | None = None,
    comparison_queries: list[str] | None = None,
    fetch_multiplier: int = 1,
) -> MagicMock:
    """Create a mock DetectionSummary with specified attributes."""
    detection = MagicMock(name="detection_summary")
    detection.query_variations = query_variations or []
    detection.comparison_entities = comparison_entities or []
    detection.comparison_queries = comparison_queries or []
    detection.fetch_multiplier = fetch_multiplier
    return detection


def _make_profile(
    query_variations: list[str] | None = None,
    comparison_entities: list[str] | None = None,
    comparison_queries: list[str] | None = None,
    fetch_multiplier: int = 1,
    top_addresses: int = 10,
    temporal_references: list[str] | None = None,
) -> RetrievalProfile:
    """Build a RetrievalProfile with detection-derived fields for testing router."""
    return RetrievalProfile(
        top_k=top_addresses * fetch_multiplier,
        top_read=top_addresses,
        query_variations=query_variations or [],
        comparison_queries=comparison_queries or [],
        comparison_entities=comparison_entities or [],
        temporal_references=temporal_references or [],
        fallback_to_chunks=False,
    )


# ---------------------------------------------------------------------------
# TestEngineDetectionInit
# ---------------------------------------------------------------------------


class TestEngineDetectionInit:
    """Tests for detection orchestrator initialization."""

    def test_detection_disabled_has_none_orchestrator(self):
        """Engine with enable_detection=False has None detection_orchestrator."""
        engine = _make_engine(enable_detection=False)

        assert engine._detection_orchestrator is None

    @patch("fitz_sage.retrieval.detection.registry.DetectionOrchestrator")
    @patch("fitz_sage.llm.factory.get_chat_factory")
    @patch("fitz_sage.llm.client.get_chat")
    @patch("fitz_sage.llm.client.get_embedder")
    @patch("fitz_sage.storage.postgres.PostgresConnectionManager")
    @patch("fitz_sage.engines.fitz_krag.ingestion.raw_file_store.RawFileStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.symbol_store.SymbolStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.import_graph_store.ImportGraphStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.section_store.SectionStore")
    @patch("fitz_sage.engines.fitz_krag.ingestion.schema.ensure_schema")
    @patch("fitz_sage.engines.fitz_krag.retrieval.strategies.code_search.CodeSearchStrategy")
    @patch("fitz_sage.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy")
    @patch("fitz_sage.engines.fitz_krag.retrieval.router.RetrievalRouter")
    @patch("fitz_sage.engines.fitz_krag.retrieval.reader.ContentReader")
    @patch("fitz_sage.engines.fitz_krag.retrieval.expander.CodeExpander")
    @patch("fitz_sage.engines.fitz_krag.query_analyzer.QueryAnalyzer")
    @patch("fitz_sage.engines.fitz_krag.context.assembler.ContextAssembler")
    @patch("fitz_sage.engines.fitz_krag.generation.synthesizer.CodeSynthesizer")
    def test_detection_enabled_creates_orchestrator(
        self,
        _synth,
        _asm,
        _qa,
        _exp,
        _reader,
        _router,
        _sect_strat,
        _code_strat,
        _ensure,
        _sect_store,
        _imp_store,
        _sym_store,
        _raw_store,
        mock_pg,
        mock_get_embedder,
        mock_get_chat,
        mock_get_chat_factory,
        mock_orchestrator_cls,
    ):
        """Engine with enable_detection=True creates a DetectionOrchestrator."""
        mock_get_embedder.return_value.dimensions = 1024
        mock_pg.get_instance.return_value = MagicMock(name="pg_instance")
        mock_factory = MagicMock(name="chat_factory")
        mock_get_chat_factory.return_value = mock_factory

        config = _make_config(enable_detection=True, enable_guardrails=False)
        engine = FitzKragEngine(config)

        # All tiers use the user-configured spec; no provider-specific
        # collapsing in the engine (legacy is_local Ollama check is gone).
        mock_get_chat_factory.assert_called_once_with(
            {
                "fast": config.chat_fast,
                "balanced": config.chat_balanced,
                "smart": config.chat_smart,
            },
            {"base_url": config.chat_base_url},
        )
        mock_orchestrator_cls.assert_called_once_with(chat_factory=mock_factory)
        assert engine._detection_orchestrator is mock_orchestrator_cls.return_value


# ---------------------------------------------------------------------------
# TestRouterDetection
# ---------------------------------------------------------------------------


class TestRouterDetection:
    """Tests for RetrievalRouter.retrieve with detection integration."""

    def _make_router(
        self,
        code_results: list[Address] | None = None,
        section_results: list[Address] | None = None,
        top_addresses: int = 10,
    ) -> RetrievalRouter:
        """Build a RetrievalRouter with mocked strategies."""
        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.return_value = code_results or []

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = section_results or []

        config = _make_config(
            top_addresses=top_addresses,
            fallback_to_chunks=False,
        )

        return RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

    def test_retrieve_without_detection_backward_compat(self):
        """Router.retrieve without detection works as before (backward compat)."""
        addr = _make_address("file.py", "foo.bar", 0.8)
        router = self._make_router(code_results=[addr])

        results = router.retrieve("what is foo?")

        assert len(results) == 1
        assert results[0].source_id == "file.py"
        # Code strategy called once (only the original query)
        router._code_strategy.retrieve.assert_called_once()

    def test_retrieve_with_query_variations_runs_extra_queries(self):
        """Router.retrieve with query_variations runs additional queries."""
        addr1 = _make_address("a.py", "mod.alpha", 0.9)
        addr2 = _make_address("b.py", "mod.beta", 0.7)
        # First call returns addr1, second returns addr2
        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.side_effect = [[addr1], [addr2]]

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []

        config = _make_config(top_addresses=10, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        profile = _make_profile(query_variations=["expanded query"])
        results = router.retrieve("original query", profile)

        # Code strategy called twice: once for "original query", once for "expanded query"
        assert code_strategy.retrieve.call_count == 2
        calls = code_strategy.retrieve.call_args_list
        assert calls[0].args[0] == "original query"
        assert calls[1].args[0] == "expanded query"
        # Both addresses returned
        assert len(results) == 2

    def test_retrieve_with_comparison_queries_from_detection(self):
        """Router.retrieve uses comparison_queries from DetectionSummary."""
        addr1 = _make_address("a.py", "mod.alpha", 0.9)

        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.return_value = [addr1]

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []

        config = _make_config(top_addresses=10, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        profile = _make_profile(
            comparison_queries=["React hooks state", "Vue composition API", "React vs Vue"],
        )
        router.retrieve("compare frameworks", profile)

        # Code strategy called 4 times: original + 3 comparison queries
        assert code_strategy.retrieve.call_count == 4
        calls = code_strategy.retrieve.call_args_list
        assert calls[0].args[0] == "compare frameworks"
        assert calls[1].args[0] == "React hooks state"
        assert calls[2].args[0] == "Vue composition API"
        assert calls[3].args[0] == "React vs Vue"

    def test_retrieve_comparison_fallback_to_entity_append(self):
        """Without comparison_queries, comparison falls back to naive entity append."""
        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.return_value = [_make_address()]

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []

        config = _make_config(top_addresses=10, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        profile = _make_profile(comparison_entities=["A", "B"])
        router.retrieve("compare A and B", profile)

        # Falls back to naive: original + "compare A and B A" + "compare A and B B"
        assert code_strategy.retrieve.call_count == 3
        calls = code_strategy.retrieve.call_args_list
        assert calls[1].args[0] == "compare A and B A"
        assert calls[2].args[0] == "compare A and B B"

    def test_retrieve_with_fetch_multiplier_increases_limit(self):
        """Router.retrieve with fetch_multiplier increases the retrieval limit."""
        addr = _make_address("a.py", "mod.func", 0.9)

        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.return_value = [addr]

        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []

        config = _make_config(top_addresses=5, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        profile = _make_profile(fetch_multiplier=3, top_addresses=5)
        router.retrieve("query", profile)

        # limit passed to code strategy should be 5 * 3 = 15
        call_args = code_strategy.retrieve.call_args
        assert call_args.args[1] == 15  # second positional arg is limit


# ---------------------------------------------------------------------------
# TestEngineAnswerDetectionFlow
# ---------------------------------------------------------------------------


class TestEngineAnswerDetectionFlow:
    """Tests for detection flowing through engine.answer() to the router."""

    def test_detection_result_flows_to_router(self):
        """Detection result from batched call is passed to router.retrieve."""
        engine = _make_engine(enable_detection=True)

        # Wire up detection orchestrator (gate_categories decides which modules run)
        mock_orchestrator = MagicMock(name="detection_orchestrator")
        mock_orchestrator.gate_categories.return_value = None  # run all modules
        mock_orchestrator._get_expansion_detector.return_value.detect.return_value = MagicMock(
            detected=False
        )
        engine._detection_orchestrator = mock_orchestrator

        # Wire up batcher to return detection results
        from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
        from fitz_sage.retrieval.detection.protocol import DetectionCategory, DetectionResult

        mock_detection_results = {
            DetectionCategory.TEMPORAL: DetectionResult.not_detected(DetectionCategory.TEMPORAL),
        }
        mock_batch_result = BatchResult(
            analysis=MagicMock(primary_type=MagicMock(value="general"), confidence=0.85),
            detection_results=mock_detection_results,
            rewrite_result=MagicMock(rewritten_query="same query", was_rewritten=False),
        )
        engine._query_batcher.batch_classify = MagicMock(return_value=mock_batch_result)

        # Wire up pipeline stages (query must trigger detection + LLM analysis — >8 words)
        query = _make_query(
            "compare the latest authentication changes between the different modules in the system"
        )
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.return_value = MagicMock(
            text="answer", provenance=[], metadata={}
        )

        engine.answer(query)

        # Batcher called (batched path)
        engine._query_batcher.batch_classify.assert_called_once()

        # Router called with a RetrievalProfile (detection data baked in)
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile

        assert isinstance(call_args[0][1], RetrievalProfile)

    def test_detection_disabled_stays_none_in_answer_flow(self):
        """When detection is disabled, detection stays None through answer flow."""
        engine = _make_engine(enable_detection=False)
        assert engine._detection_orchestrator is None

        # Wire up pipeline stages (query must be >8 words to trigger LLM analysis)
        query = _make_query(
            "How does the authentication system work when handling multiple concurrent user sessions?"
        )
        engine._retrieval_router.retrieve.return_value = [MagicMock()]
        engine._reader.read.return_value = [MagicMock()]
        engine._expander.expand.return_value = [MagicMock()]
        engine._assembler.assemble.return_value = MagicMock()
        engine._synthesizer.generate.return_value = MagicMock(
            text="answer", provenance=[], metadata={}
        )

        engine.answer(query)

        # Router called with query text and a profile built from detection=None
        engine._retrieval_router.retrieve.assert_called_once()
        call_args = engine._retrieval_router.retrieve.call_args
        assert call_args[0][0] == query.text
        from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile

        profile = call_args[0][1]
        assert isinstance(profile, RetrievalProfile)
        # No detection ran: profile carries no detection-derived query expansions
        assert profile.query_variations == []
        assert profile.comparison_queries == []


# ---------------------------------------------------------------------------
# TestTemporalTagging
# ---------------------------------------------------------------------------


class TestTemporalTagging:
    """Tests for temporal reference tagging on addresses."""

    def _make_router(
        self,
        code_results: list[Address] | None = None,
        top_addresses: int = 10,
    ) -> RetrievalRouter:
        code_strategy = MagicMock(name="code_strategy")
        code_strategy.retrieve.return_value = code_results or []
        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []
        config = _make_config(top_addresses=top_addresses, fallback_to_chunks=False)
        return RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

    def test_temporal_variations_tag_addresses(self):
        """Addresses from temporal query variations get tagged with temporal_refs."""
        addr = _make_address("a.py", "report.revenue", 0.8)
        router = self._make_router(code_results=[addr])

        profile = _make_profile(
            query_variations=["Q1 2024 revenue", "Q2 2024 revenue"],
            temporal_references=["Q1 2024", "Q2 2024"],
        )

        results = router.retrieve("compare Q1 vs Q2 revenue", profile)

        # Addresses from temporal queries should have temporal_refs in metadata
        tagged = [r for r in results if r.metadata.get("temporal_refs")]
        assert len(tagged) > 0
        # The deduplicated address should have both refs merged
        all_refs = tagged[0].metadata["temporal_refs"]
        assert "Q1 2024" in all_refs
        assert "Q2 2024" in all_refs

    def test_original_query_has_no_temporal_tag(self):
        """Addresses from the original query should NOT get temporal tags."""
        addr = _make_address("a.py", "unique.symbol", 0.9)
        code_strategy = MagicMock(name="code_strategy")
        # First call (original) returns addr, subsequent calls return different addr
        code_strategy.retrieve.side_effect = [
            [addr],
            [_make_address("b.py", "other.symbol", 0.7)],
        ]
        section_strategy = MagicMock(name="section_strategy")
        section_strategy.retrieve.return_value = []
        config = _make_config(top_addresses=10, fallback_to_chunks=False)
        router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=config,
            section_strategy=section_strategy,
        )

        profile = _make_profile(
            query_variations=["Q1 revenue"],
            temporal_references=["Q1 2024"],
        )

        results = router.retrieve("revenue trends", profile)

        # Find the original query's address (unique.symbol)
        original = [r for r in results if r.location == "unique.symbol"]
        assert len(original) == 1
        assert "temporal_refs" not in original[0].metadata

    def test_no_temporal_refs_when_detection_has_no_temporal(self):
        """Without temporal in detection, no temporal_refs are added."""
        addr = _make_address("a.py", "mod.func", 0.9)
        router = self._make_router(code_results=[addr])

        profile = _make_profile(
            query_variations=["synonym query"],
            temporal_references=[],  # No temporal data
        )

        results = router.retrieve("original query", profile)

        for r in results:
            assert "temporal_refs" not in r.metadata


# ---------------------------------------------------------------------------
# TestDeduplicateTemporalMerge
# ---------------------------------------------------------------------------


class TestDeduplicateTemporalMerge:
    """Tests for _deduplicate merging temporal_refs from duplicates."""

    def test_merges_temporal_refs_on_dedup(self):
        """When same address appears with different temporal tags, tags are merged."""
        router = RetrievalRouter(
            code_strategy=MagicMock(),
            chunk_strategy=None,
            config=_make_config(top_addresses=10),
        )

        addr_q1 = Address(
            kind=AddressKind.SYMBOL,
            source_id="f.py",
            location="report.revenue",
            summary="Revenue report",
            score=0.8,
            metadata={"temporal_refs": ["Q1 2024"]},
        )
        addr_q2 = Address(
            kind=AddressKind.SYMBOL,
            source_id="f.py",
            location="report.revenue",
            summary="Revenue report",
            score=0.9,
            metadata={"temporal_refs": ["Q2 2024"]},
        )

        deduped = router._deduplicate([addr_q1, addr_q2])

        assert len(deduped) == 1
        assert set(deduped[0].metadata["temporal_refs"]) == {"Q1 2024", "Q2 2024"}
        # Takes the higher score
        assert deduped[0].score == 0.9

    def test_dedup_without_temporal_refs_unchanged(self):
        """Deduplication without temporal refs works as before."""
        router = RetrievalRouter(
            code_strategy=MagicMock(),
            chunk_strategy=None,
            config=_make_config(top_addresses=10),
        )

        addr1 = _make_address("a.py", "mod.func", 0.8)
        addr2 = _make_address("a.py", "mod.func", 0.6)

        deduped = router._deduplicate([addr1, addr2])

        assert len(deduped) == 1
        assert deduped[0].score == 0.8  # First one kept
