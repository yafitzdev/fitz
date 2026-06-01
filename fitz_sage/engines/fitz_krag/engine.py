# fitz_sage/engines/fitz_krag/engine.py
"""
FitzKragEngine - Knowledge Routing Augmented Generation engine.

Uses knowledge-type-aware access strategies (code symbols, document sections)
instead of uniform chunk-based retrieval. Retrieval returns addresses (pointers),
content is read on demand after ranking.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fitz_sage.core import (
    Answer,
    ConfigurationError,
    EvidenceItem,
    EvidencePack,
    GenerationError,
    KnowledgeError,
    Query,
    QueryError,
)
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis
    from fitz_sage.engines.fitz_krag.types import Address, ReadResult

logger = get_logger(__name__)


def _report_timings(
    progress: Callable[[str], None],
    timings: list[tuple[str, float]],
    pipeline_start: float,
) -> None:
    """Report pipeline timing breakdown via progress callback."""
    import time

    total = time.perf_counter() - pipeline_start
    parts = "  ".join(f"{name}: {dur:.1f}s" for name, dur in timings)
    progress(f"Pipeline: {total:.1f}s total — {parts}")


def _build_provider_config(
    base_url: str | None,
    api_key_env: str | None,
    *,
    spec: str | None = None,
    auth: dict[str, Any] | None = None,
    cert_path: str | None = None,
) -> dict[str, Any] | None:
    """
    Build a config dict for ``get_chat`` / ``get_vision``.

    Only includes ``base_url`` when the provider spec consumes it. The
    ``openai`` preset has its own default URL — overriding it from the
    engine schema's local-default base_url would silently route cloud
    calls to localhost. Auth is forwarded for every provider because it
    may contain endpoint API keys, M2M OAuth2, or enterprise composite
    credentials.
    """
    if spec is None:
        # Caller doesn't know the spec — be permissive (used by callers
        # that already know the spec is endpoint-family).
        consumes_base_url = base_url is not None
    else:
        provider = spec.split("/", 1)[0].strip()
        # endpoint always uses base_url; enterprise requires it; the
        # openai/azure_openai presets accept it as a proxy override
        # but we only forward it when the user explicitly opted in
        # (i.e. set the field non-default), which we can't distinguish
        # from defaults here — so for openai/azure_openai we omit it.
        consumes_base_url = provider in ("endpoint", "enterprise")

    cfg: dict[str, Any] = {}
    if consumes_base_url and base_url is not None:
        cfg["base_url"] = base_url
    if cert_path is not None:
        cfg["cert_path"] = cert_path
    if auth is not None:
        auth_cfg = dict(auth)
        if api_key_env is not None and "type" not in auth_cfg:
            auth_cfg.setdefault("api_key_env", api_key_env)
        cfg["auth"] = auth_cfg
    elif api_key_env is not None:
        cfg["auth"] = {"api_key_env": api_key_env}

    return cfg or None


@dataclass
class _RetrievalOutcome:
    """Carrier for the retrieval half of the KRAG pipeline.

    Produced by ``_retrieve_core``; consumed by ``answer()`` and ``retrieve()``.
    """

    sanitized: str
    expanded: list[ReadResult]
    addresses: list[Address]
    timings: list[tuple[str, float]]


class FitzKragEngine:
    """
    Fitz KRAG engine implementation.

    Flow:
    1. Analyze query intent (+ optional detection)
    2. Retrieve addresses (pointers to code symbols / document sections)
    3. Read content for top-ranked addresses
    4. Expand with context (imports, class context, same-file refs)
    5. Run epistemic governance — determine AnswerMode
    6. Assemble LLM context
    7. Generate grounded answer with file:line provenance
    """

    def __init__(self, config: FitzKragConfig):
        try:
            self._config = config
            self._bg_worker: Any = None
            self._manifest: Any = None
            self._source_dir: Path | None = None
            self._init_components()
        except Exception as e:
            msg = str(e)
            if "ConnectError" in type(e).__name__ or "10061" in msg or "Connection refused" in msg:
                raise ConfigurationError(
                    "Cannot connect to the configured endpoint chat provider.\n"
                    "Required enrichment uses Fitz's managed local Qwen ONNX runtime; "
                    "check chat_base_url only for optional endpoint synthesis."
                ) from e
            raise ConfigurationError(f"Failed to initialize Fitz KRAG engine: {e}") from e

    def load(self, collection: str) -> None:
        """Load a collection, reinitializing collection-dependent components."""
        # Stop background worker when switching collections
        if self._bg_worker and collection != self._config.collection:
            self._bg_worker.stop()
            self._bg_worker = None
            self._manifest = None
            self._source_dir = None

        self._config.collection = collection
        self._init_components()

        # Re-wire progressive state if still active after reload
        if self._manifest and self._source_dir:
            self._wire_agentic_strategy()
        else:
            # Auto-detect persisted manifest from a previous `point()` call
            self._try_load_persisted_manifest(collection)

    def _wire_agentic_strategy(self) -> None:
        """Wire agentic strategy + disk fallback from current manifest/source_dir."""
        from fitz_sage.core.paths import FitzPaths
        from fitz_sage.engines.fitz_krag.retrieval.strategies.agentic_search import (
            AgenticSearchStrategy,
        )

        col_dir = FitzPaths.workspace() / "collections" / self._config.collection
        agentic = AgenticSearchStrategy(
            manifest=self._manifest,
            source_dir=self._source_dir,
            chat_factory=self._chat_factory,
            config=self._config,
            cache_dir=col_dir / "parsed",
        )
        self._retrieval_router._agentic_strategy = agentic
        self._retrieval_router._allow_llm_agentic = self._chat_factory is not None
        self._reader._source_dir = self._source_dir

    def _try_load_persisted_manifest(self, collection: str) -> None:
        """Load manifest + source_dir from disk if they exist from a prior point() call."""
        from fitz_sage.core.paths import FitzPaths

        col_dir = FitzPaths.workspace() / "collections" / collection
        manifest_path = col_dir / "manifest.json"
        source_dir_path = col_dir / "source_dir.txt"

        if not manifest_path.exists() or not source_dir_path.exists():
            return

        try:
            from fitz_sage.engines.fitz_krag.progressive.manifest import FileManifest

            source_dir = Path(source_dir_path.read_text(encoding="utf-8").strip())
            if not source_dir.exists():
                logger.debug(f"Persisted source_dir no longer exists: {source_dir}")
                return

            self._manifest = FileManifest(manifest_path)
            self._source_dir = source_dir
            self._wire_agentic_strategy()
            logger.info(
                f"Loaded manifest for collection '{collection}' ({len(self._manifest.entries())} files)"
            )
        except Exception as e:
            logger.debug(f"Failed to load persisted manifest: {e}")

    def _init_components(self) -> None:
        """Initialize engine components lazily."""
        import time as _t

        from fitz_sage.storage.sqlite import SqliteConnectionManager

        _t0 = _t.perf_counter()

        chat_config = _build_provider_config(
            self._config.chat_base_url,
            self._config.chat_api_key_env,
            spec=self._config.chat_smart,
            auth=self._config.auth,
            cert_path=self._config.cert_path,
        )

        logger.debug("Starting database")
        self._chat = None
        self._connection_manager = SqliteConnectionManager.get_instance()
        logger.debug("Database initialized")

        _t1 = _t.perf_counter()
        logger.debug(f"[init] providers+pg: {(_t1-_t0)*1000:.0f}ms")

        # Ingestion stores
        from fitz_sage.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
        from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
        from fitz_sage.engines.fitz_krag.ingestion.schema import ensure_schema
        from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore
        from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore

        self._raw_store = RawFileStore(self._connection_manager, self._config.collection)
        self._symbol_store = SymbolStore(self._connection_manager, self._config.collection)
        self._import_store = ImportGraphStore(self._connection_manager, self._config.collection)
        self._section_store = SectionStore(self._connection_manager, self._config.collection)

        # Table stores
        from fitz_sage.engines.fitz_krag.ingestion.table_store import TableStore
        from fitz_sage.tabular.store.sqlite import SqliteTableStore

        self._table_store = TableStore(self._connection_manager, self._config.collection)
        self._sqlite_table_store = SqliteTableStore(self._config.collection)

        _ts1 = _t.perf_counter()
        logger.debug(f"[init] store objects: {(_ts1-_t1)*1000:.0f}ms")

        # Retrieval
        from fitz_sage.engines.fitz_krag.retrieval.expander import CodeExpander
        from fitz_sage.engines.fitz_krag.retrieval.reader import ContentReader
        from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
        from fitz_sage.engines.fitz_krag.retrieval.strategies.code_search import (
            CodeSearchStrategy,
        )
        from fitz_sage.engines.fitz_krag.retrieval.strategies.section_search import (
            SectionSearchStrategy,
        )
        from fitz_sage.engines.fitz_krag.retrieval.strategies.table_search import (
            TableSearchStrategy,
        )

        code_strategy = CodeSearchStrategy(self._symbol_store, self._config)
        section_strategy = SectionSearchStrategy(self._section_store, self._config)
        table_strategy = TableSearchStrategy(self._table_store, self._config)
        self._retrieval_router = RetrievalRouter(
            code_strategy=code_strategy,
            config=self._config,
            section_strategy=section_strategy,
            table_strategy=table_strategy,
        )
        self._reader = ContentReader(
            self._raw_store,
            section_store=self._section_store,
            config=self._config,
            table_store=self._table_store,
            sqlite_table_store=self._sqlite_table_store,
        )
        self._expander = CodeExpander(
            self._raw_store,
            self._symbol_store,
            self._import_store,
            self._config,
        )

        _t3 = _t.perf_counter()
        logger.debug(f"[init] strategies: {(_t3-_ts1)*1000:.0f}ms")

        # Chat factory for legacy tiered LLM paths. Retrieval-first defaults
        # leave chat tiers unset, so this stays None unless the user opts in.
        from fitz_sage.llm.factory import get_chat_factory

        tier_specs = self._chat_tier_specs()
        self._chat_factory = (
            get_chat_factory(tier_specs, chat_config) if tier_specs is not None else None
        )

        # Context + Generation
        from fitz_sage.engines.fitz_krag.context.assembler import ContextAssembler
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer
        from fitz_sage.llm.client import get_chat
        from fitz_sage.llm.providers.onnx_chat import OnnxChat

        self._assembler = ContextAssembler(self._config)
        self._synthesizer = None
        if self._config.synthesizer:
            synth_config = _build_provider_config(
                self._config.chat_base_url,
                self._config.chat_api_key_env,
                spec=self._config.synthesizer,
                auth=self._config.auth,
                cert_path=self._config.cert_path,
            )
            synth_chat = get_chat(self._config.synthesizer, "smart", synth_config)
            self._synthesizer = CodeSynthesizer(synth_chat, self._config)

        standard_chat = OnnxChat()
        self._enricher_chat = standard_chat
        self._summarizer_chat = standard_chat

        # Governance — pyrrho classifier (single INT8 ONNX forward pass).
        # Provider-presence: the `governance:` config key builds the
        # classifier (`pyrrho` / `pyrrho/<model>`) or disables it (`null`).
        # The model lazily loads on first decide() so engine init stays fast.
        from fitz_sage.governance import create_governance

        self._governance = create_governance(self._config.governance)

        # Table query handler
        from fitz_sage.engines.fitz_krag.retrieval.table_handler import TableQueryHandler

        table_chat = self._chat_factory("balanced") if self._chat_factory else None
        self._table_handler = TableQueryHandler(
            table_chat,
            self._sqlite_table_store,
            self._config,
        )

        # Query prep defaults to the deterministic planner. If
        # query_intelligence is configured, the batcher uses that provider as
        # an optional LLM enhancer.
        from fitz_sage.engines.fitz_krag.query_batcher import QueryBatcher
        from fitz_sage.engines.fitz_krag.query_planner import DeterministicQueryPlanner
        from fitz_sage.retrieval.detection.modules import DEFAULT_MODULES

        query_chat_factory = self._chat_factory or self._missing_chat_factory
        if self._config.query_intelligence:
            query_chat_config = _build_provider_config(
                self._config.chat_base_url,
                self._config.chat_api_key_env,
                spec=self._config.query_intelligence,
                auth=self._config.auth,
                cert_path=self._config.cert_path,
            )
            query_chat_factory = get_chat_factory(
                {
                    "fast": self._config.query_intelligence,
                    "balanced": self._config.query_intelligence,
                    "smart": self._config.query_intelligence,
                },
                query_chat_config,
            )
        self._query_planner = DeterministicQueryPlanner()
        self._query_batcher = QueryBatcher(
            chat_factory=query_chat_factory,
            detection_modules=list(DEFAULT_MODULES),
        )

        # Reranker — INT8 ONNX cross-encoder via get_reranker().
        # Default backbone: Alibaba-NLP/gte-reranker-modernbert-base.
        # Override with rerank: "onnx/<hf-model-id>" or disable with null.
        self._address_reranker: Any = None
        if self._config.rerank:
            from fitz_sage.llm.client import get_reranker

            reranker = get_reranker(self._config.rerank)

            if reranker:
                from fitz_sage.engines.fitz_krag.retrieval.reranker import AddressReranker

                self._address_reranker = AddressReranker(
                    reranker=reranker,
                    k=self._config.rerank_k,
                    min_addresses=self._config.rerank_min_addresses,
                )

        # LLM structural code search (default when chat available)
        if self._config.code_search_mode != "hybrid" and self._chat_factory:
            from fitz_sage.engines.fitz_krag.retrieval.strategies.llm_code_search import (
                LlmCodeSearchStrategy,
            )

            llm_strategy = LlmCodeSearchStrategy(
                symbol_store=self._symbol_store,
                import_store=self._import_store,
                chat_factory=self._chat_factory,
                config=self._config,
                fallback_strategy=code_strategy,
            )
            self._retrieval_router._code_strategy = llm_strategy
            code_strategy = llm_strategy  # so HyDE/raw_store wiring below reaches it

        # Wire raw_store for freshness boosting
        code_strategy._raw_store = self._raw_store
        section_strategy._raw_store = self._raw_store

        # Entity graph store
        self._entity_graph_store: Any = None
        try:
            from fitz_sage.retrieval.entity_graph.store import EntityGraphStore

            self._entity_graph_store = EntityGraphStore(collection=self._config.collection)
            self._expander._entity_graph_store = self._entity_graph_store
        except Exception as e:
            logger.debug(f"Entity graph store init: {e}")

        # Retrieval pass — Tiers 1-4 (retrieve -> rerank -> read) as one unit.
        from fitz_sage.engines.fitz_krag.retrieval.retrieval_pass import RetrievalPass

        self._retrieval_pass = RetrievalPass(
            router=self._retrieval_router,
            reranker=self._address_reranker,
            reader=self._reader,
            config=self._config,
        )

        # Multi-hop controller — loops the retrieval pass, pyrrho-gated.
        self._hop_controller: Any = None
        if self._config.enable_multi_hop and self._chat_factory:
            from fitz_sage.engines.fitz_krag.retrieval.multihop import KragHopController

            self._hop_controller = KragHopController(
                retrieval_pass=self._retrieval_pass,
                chat_factory=self._chat_factory,
                governance=self._governance,
                max_hops=self._config.max_hops,
            )

        _t4 = _t.perf_counter()
        logger.debug(f"[init] components: {(_t4-_t3)*1000:.0f}ms")

        ensure_schema(self._connection_manager, self._config.collection)

        _t5 = _t.perf_counter()
        logger.debug(f"[init] schema: {(_t5-_t4)*1000:.0f}ms, " f"total: {(_t5-_t0)*1000:.0f}ms")

        # Retrieval-first init intentionally does not create or warm chat clients.

    def _chat_tier_specs(self) -> dict[str, str] | None:
        """Return complete tier specs when the legacy chat tiers are configured."""
        configured = [
            spec
            for spec in (
                self._config.chat_fast,
                self._config.chat_balanced,
                self._config.chat_smart,
            )
            if spec
        ]
        if not configured:
            return None

        fallback = configured[0]
        return {
            "fast": self._config.chat_fast or fallback,
            "balanced": self._config.chat_balanced or fallback,
            "smart": self._config.chat_smart or fallback,
        }

    @staticmethod
    def _missing_chat_factory(tier: str = "fast") -> Any:
        """Raise when an LLM enhancement is requested without a provider."""
        raise ConfigurationError(
            f"No chat provider configured for tier '{tier}'. Configure the specific "
            "feature provider, or set chat_fast/chat_balanced/chat_smart."
        )

    # Keywords that signal the query may have temporal/comparison/aggregation intent.
    # If none match, the detection LLM call can be skipped safely.
    _DETECTION_KEYWORDS = frozenset(
        [
            # Temporal
            "latest",
            "recent",
            "last",
            "before",
            "after",
            "since",
            "until",
            "new",
            "old",
            "updated",
            "changed",
            "history",
            "previous",
            # Comparison
            "vs",
            "versus",
            "compare",
            "differ",
            "difference",
            "between",
            "better",
            "worse",
            "advantage",
            "disadvantage",
            # Aggregation
            "how many",
            "count",
            "list",
            "all",
            "every",
            "enumerate",
            "total",
            "summarize",
            "overview",
            # Freshness
            "current",
            "now",
            "today",
        ]
    )

    @staticmethod
    def _needs_detection(query: str) -> bool:
        """Return True if query may benefit from LLM detection.

        Short, simple queries without temporal/comparison/aggregation
        keywords won't trigger any detection module, so we can skip the
        LLM call entirely.
        """
        words = query.lower().split()
        n = len(words)

        # Long or complex queries: always run detection
        if n > 10:
            return True

        # Check for detection-triggering keywords
        query_lower = query.lower()
        for kw in FitzKragEngine._DETECTION_KEYWORDS:
            if kw in query_lower:
                return True

        return False

    # Words to ignore when extracting entities from a query.
    _STOP_WORDS = frozenset(
        "what where who when how is are does do did the a an of in to for on "
        "with by from about it this that these those my your its can could "
        "should would will shall may might be been being have has had".split()
    )

    def _build_detection_summary(self, results: dict) -> Any:
        """Wrap batched detection-module results into a DetectionSummary."""
        from fitz_sage.retrieval.detection.protocol import (
            DetectionCategory,
            DetectionResult,
        )
        from fitz_sage.retrieval.detection.registry import DetectionSummary

        return DetectionSummary(
            temporal=results.get(
                DetectionCategory.TEMPORAL,
                DetectionResult.not_detected(DetectionCategory.TEMPORAL),
            ),
            aggregation=results.get(
                DetectionCategory.AGGREGATION,
                DetectionResult.not_detected(DetectionCategory.AGGREGATION),
            ),
            comparison=results.get(
                DetectionCategory.COMPARISON,
                DetectionResult.not_detected(DetectionCategory.COMPARISON),
            ),
            freshness=results.get(
                DetectionCategory.FRESHNESS,
                DetectionResult.not_detected(DetectionCategory.FRESHNESS),
            ),
        )

    @staticmethod
    def _fast_analyze(query: str) -> "QueryAnalysis | None":
        """Try to classify simple queries without an LLM call.

        Returns QueryAnalysis for short, straightforward queries where LLM
        classification adds no value. Returns None for complex queries that
        need LLM analysis.
        """
        from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType

        words = query.split()
        n = len(words)

        # Complex queries always need LLM analysis
        if n > 8:
            return None

        # Extract entities: non-stop content words (preserving original case)
        entities = tuple(
            w.rstrip("?.,!;:")
            for w in words
            if w.lower().rstrip("?.,!;:") not in FitzKragEngine._STOP_WORDS and len(w) > 1
        )

        return QueryAnalysis(
            primary_type=QueryType.GENERAL,
            confidence=0.9,
            entities=entities,
            refined_query=query,
        )

    def answer(self, query: Query, *, progress: Callable[[str], None] | None = None) -> Answer:
        """
        Execute a query using KRAG approach.

        Flow: analyze → detect → retrieve → (cache check) → read → expand →
              governance → assemble → generate → (cache store)

        Args:
            query: Query object with question text
            progress: Optional callback for status updates (e.g. ui.info)

        Returns:
            Answer with file:line provenance
        """
        import time

        from fitz_sage.engines.fitz_krag.context.compressor import compress_results

        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")
        if self._synthesizer is None:
            raise GenerationError(
                "No synthesizer configured. Run `fitz retrieve ...` for evidence, "
                "or configure a synthesizer provider."
            )

        with self._query_scope():
            logger.info(f"Starting query processing (query_length={len(query.text)})")
            try:
                _progress = progress or (lambda _: None)
                pipeline_start = time.perf_counter()

                # 1-4. Analyze, detect, retrieve, read, expand, table queries
                outcome = self._retrieve_core(query, progress=progress)
                sanitized = outcome.sanitized
                expanded = outcome.expanded
                timings = outcome.timings

                if not expanded:
                    _report_timings(_progress, timings, pipeline_start)
                    gap_context = self._build_gap_context(sanitized)
                    return Answer(
                        text=self._synthesizer._build_abstain_message(sanitized, gap_context),
                        provenance=[],
                        mode=AnswerMode.ABSTAIN,
                        metadata={
                            "engine": "fitz_krag",
                            "query": query.text,
                            "answer_mode": "abstain",
                            "gap_context": gap_context,
                        },
                    )

                # 5. Run governance — pyrrho classifier on the (query, contexts) pair.
                # ReadResult satisfies EvidenceItem (has .content).
                answer_mode = AnswerMode.TRUSTWORTHY
                governance = None
                if self._governance is not None:
                    t0 = time.perf_counter()
                    governance = self._governance.decide(sanitized, expanded)
                    answer_mode = governance.mode

                    # Progressive mode: agentic LLM code search already validated
                    # relevance structurally. If pyrrho returns ABSTAIN on a result
                    # set produced by the progressive manifest path, prefer to
                    # generate — the structural retrieval is the stronger signal
                    # here than the classifier's distribution-shifted call.
                    if (
                        answer_mode == AnswerMode.ABSTAIN
                        and self._manifest is not None
                        and expanded
                    ):
                        logger.info(
                            "Overriding ABSTAIN -> TRUSTWORTHY in progressive mode "
                            "(structural retrieval validated relevance)"
                        )
                        answer_mode = AnswerMode.TRUSTWORTHY

                    timings.append(("Governance", time.perf_counter() - t0))

                # 5.5. Compress code context (AST-based, ~50-70% token reduction)
                expanded = compress_results(expanded)

                # 6. Assemble context
                context = self._assembler.assemble(sanitized, expanded)

                # 7. Generate answer with answer mode
                _progress("Generating answer...")
                t0 = time.perf_counter()
                gap_context = None
                conflict_context = None
                if answer_mode == AnswerMode.ABSTAIN:
                    governance_reasons = governance.reasons if governance else ()
                    gap_context = self._build_gap_context(sanitized, governance_reasons)
                elif answer_mode == AnswerMode.DISPUTED and governance:
                    conflict_context = {"reason": governance.reason}
                answer = self._synthesizer.generate(
                    sanitized,
                    context,
                    expanded,
                    answer_mode=answer_mode,
                    gap_context=gap_context,
                    conflict_context=conflict_context,
                )
                timings.append(("Generation", time.perf_counter() - t0))

                # Report timing breakdown
                _report_timings(_progress, timings, pipeline_start)

                # 7.5. Flag queried files for background-worker priority + warming
                self._boost_queried_files(outcome)

                return answer

            except Exception as e:
                error_msg = str(e).lower()
                if "retriev" in error_msg or "search" in error_msg:
                    raise KnowledgeError(f"Retrieval failed: {e}") from e
                elif "generat" in error_msg or "llm" in error_msg:
                    raise GenerationError(f"Generation failed: {e}") from e
                else:
                    raise KnowledgeError(f"KRAG pipeline error: {e}") from e

    def retrieve(
        self, query: Query, *, progress: Callable[[str], None] | None = None
    ) -> list[ReadResult]:
        """Retrieve relevant sources for a query — content, no synthesis.

        Runs the KRAG retrieval pipeline (analyze, detect, retrieve, read,
        expand, compress) and returns the expanded, compressed results. This
        is the retrieval primitive ``answer()`` builds on; use it directly
        when the source material is wanted rather than a generated answer.

        Args:
            query: Query object with question text.
            progress: Optional callback for status updates.

        Returns:
            List of ReadResult with file content and Address provenance.
            Empty list when nothing relevant is found.
        """
        from fitz_sage.engines.fitz_krag.context.compressor import compress_results

        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        with self._query_scope():
            logger.info(f"Starting retrieval (query_length={len(query.text)})")
            try:
                outcome = self._retrieve_core(query, progress=progress)
                self._boost_queried_files(outcome)
                if not outcome.expanded:
                    return []
                return compress_results(outcome.expanded)
            except Exception as e:
                raise KnowledgeError(f"Retrieval failed: {e}") from e

    def evidence(
        self,
        query: Query,
        *,
        progress: Callable[[str], None] | None = None,
        top_k: int | None = None,
    ) -> EvidencePack:
        """Return a governed evidence pack for a query, without synthesis or chat prep."""
        import time

        from fitz_sage.engines.fitz_krag.context.compressor import compress_results

        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        with self._query_scope():
            logger.info(f"Starting evidence retrieval (query_length={len(query.text)})")
            try:
                outcome = self._retrieve_core(
                    query,
                    progress=progress,
                    use_query_intelligence=False,
                    allow_llm_strategies=False,
                    execute_table_queries=False,
                )
                expanded = compress_results(outcome.expanded) if outcome.expanded else []
                timings = list(outcome.timings)

                mode: AnswerMode | None = None
                reasons: list[str] = []
                if not expanded:
                    mode = AnswerMode.ABSTAIN
                    reasons = ["No relevant evidence retrieved."]
                elif self._governance is not None:
                    t0 = time.perf_counter()
                    governance = self._governance.decide(outcome.sanitized, expanded)
                    timings.append(("Governance", time.perf_counter() - t0))
                    mode = governance.mode
                    reasons = list(governance.reasons)

                requested_top_k = top_k or query.metadata.get("top_k")
                items = self._build_evidence_items(expanded)
                if requested_top_k is not None:
                    items = items[: int(requested_top_k)]

                self._boost_queried_files(outcome)

                return EvidencePack(
                    query=outcome.sanitized,
                    mode=mode,
                    items=items,
                    reasons=reasons,
                    timings={name: duration for name, duration in timings},
                    indexing_status=self.indexing_status(),
                    metadata={"engine": "fitz_krag", "source_query": query.text},
                )
            except Exception as e:
                raise KnowledgeError(f"Evidence retrieval failed: {e}") from e

    def _build_evidence_items(self, results: list["ReadResult"]) -> list[EvidenceItem]:
        """Convert KRAG read results into stable core evidence items."""
        items: list[EvidenceItem] = []
        for rank, result in enumerate(results, start=1):
            address = result.address
            kind = getattr(address.kind, "value", str(address.kind))
            metadata = {**address.metadata, **result.metadata}
            items.append(
                EvidenceItem(
                    rank=rank,
                    source_id=address.source_id,
                    file_path=result.file_path,
                    address_kind=kind,
                    address_location=address.location,
                    line_range=result.line_range,
                    score=address.score,
                    excerpt=self._excerpt(result.content),
                    content=result.content,
                    metadata=metadata,
                )
            )
        return items

    @staticmethod
    def _excerpt(content: str, max_chars: int = 320) -> str:
        """Build a compact one-line excerpt."""
        import re

        text = re.sub(r"\s+", " ", content).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _boost_queried_files(self, outcome: "_RetrievalOutcome") -> None:
        """Flag the files this query surfaced for the background worker.

        Bumps them to P1 so the worker prioritizes their eager indexing and,
        after the eager phases, summarizes them on demand (the warm loop).
        Agentic results carry the rel_path in metadata; section/code/table
        results carry the file id in ``source_id``, resolved via the manifest.
        """
        if not self._bg_worker or not self._manifest:
            return
        rel_by_id = {e.file_id: rp for rp, e in self._manifest.entries().items()}
        rel_paths: set[str] = set()
        for addr in outcome.addresses:
            disk_path = addr.metadata.get("disk_path")
            if disk_path:
                rel_paths.add(disk_path)
            elif addr.source_id in rel_by_id:
                rel_paths.add(rel_by_id[addr.source_id])
        if rel_paths:
            self._bg_worker.boost_files(list(rel_paths))

    @contextmanager
    def _query_scope(self) -> Any:
        """Query-scoped logging context + background-worker signalling.

        Shared by answer() and retrieve() — both are queries from the
        background worker's perspective.
        """
        import uuid

        from fitz_sage.logging import clear_query_context, set_query_context

        set_query_context(query_id=f"q-{uuid.uuid4().hex[:8]}")
        if self._bg_worker:
            self._bg_worker.signal_query_start()
        try:
            yield
        finally:
            if self._bg_worker:
                self._bg_worker.signal_query_end()
            clear_query_context()

    @contextmanager
    def _retrieval_strategy_scope(self, allow_llm_strategies: bool) -> Any:
        """Temporarily force deterministic retrieval strategies for evidence mode."""
        if allow_llm_strategies:
            yield
            return

        router = self._retrieval_router
        original_code_strategy = getattr(router, "_code_strategy", None)
        original_allow_agentic = getattr(router, "_allow_llm_agentic", True)

        fallback = getattr(original_code_strategy, "_fallback", None)
        if fallback is not None:
            router._code_strategy = fallback
        if hasattr(router, "_allow_llm_agentic"):
            router._allow_llm_agentic = False

        try:
            yield
        finally:
            if original_code_strategy is not None:
                router._code_strategy = original_code_strategy
            if hasattr(router, "_allow_llm_agentic"):
                router._allow_llm_agentic = original_allow_agentic

    def _retrieve_core(
        self,
        query: Query,
        *,
        progress: Callable[[str], None] | None = None,
        use_query_intelligence: bool | None = None,
        allow_llm_strategies: bool = True,
        execute_table_queries: bool = True,
    ) -> _RetrievalOutcome:
        """Run the retrieval half of the KRAG pipeline.

        Analyze + detect → retrieve → read → expand → table queries. Returns
        the expanded ReadResults (pre-governance, pre-compression). Callers
        guarantee non-empty query text and supply the query-scoped context.
        """
        import re
        import time

        from fitz_sage.engines.fitz_krag.query_planner import (
            DeterministicQueryPlanner,
            plan_from_batch_result,
        )
        from fitz_sage.engines.fitz_krag.retrieval_profile import build_retrieval_profile

        # 0. Sanitize and normalize query
        sanitized = re.sub(r"<[^>]+>", "", query.text).strip()
        if not sanitized:
            sanitized = query.text.strip()

        # Truncate only pathologically long input — multi-query
        # decomposition handles genuinely long queries downstream.
        MAX_QUERY_LENGTH = 8000
        if len(sanitized) > MAX_QUERY_LENGTH:
            original_length = len(sanitized)
            sanitized = sanitized[:MAX_QUERY_LENGTH]
            logger.debug(
                "Query truncated", original_length=original_length, new_length=MAX_QUERY_LENGTH
            )

        _progress = progress or (lambda _: None)
        timings: list[tuple[str, float]] = []

        _progress("Analyzing query...")
        t0 = time.perf_counter()

        if use_query_intelligence is None:
            use_query_intelligence = self._config.query_intelligence is not None

        planner = getattr(self, "_query_planner", None) or DeterministicQueryPlanner()
        plan = planner.plan(sanitized, detection_enabled=True)

        if use_query_intelligence:
            # Query prep — one batched LLM call: rewrite + analysis +
            # detection + keywords. Optional enhancement over the no-chat plan.
            fast_analysis = self._fast_analyze(sanitized)
            need_llm_analysis = fast_analysis is None
            need_detection = self._needs_detection(sanitized)

            try:
                batch_result = self._query_batcher.batch_classify(
                    sanitized,
                    include_analysis=need_llm_analysis,
                    include_detection=need_detection,
                    include_rewriting=True,
                    include_extended=True,
                    include_keywords=True,
                    conversation_context=query.metadata.get("conversation_context"),
                )
                llm_detection = (
                    self._build_detection_summary(batch_result.detection_results)
                    if need_detection and batch_result.detection_results is not None
                    else plan.detection
                )
                plan = plan_from_batch_result(
                    sanitized,
                    batch_result,
                    fallback_analysis=fast_analysis or plan.analysis,
                    detection=llm_detection,
                    fallback_plan=plan,
                )
                if plan.rewrite_result and plan.retrieval_query != sanitized:
                    logger.debug(
                        "Query rewritten",
                        original_preview=sanitized[:50],
                        rewritten_preview=plan.retrieval_query[:50],
                    )
            except Exception as e:
                logger.warning(f"Batched query intelligence failed: {e}")
        timings.append(("Query prep", time.perf_counter() - t0))

        # Build retrieval profile — single object with all gates and signals
        profile = build_retrieval_profile(
            plan.analysis,
            plan.detection,
            self._config,
            extended_signals=plan.extended_signals,
            keywords=plan.keywords,
        )

        # 2. Retrieve — one retrieval pass, or a multi-hop loop of passes.
        #    A pass is Tiers 1-4: retrieve -> fuse -> rerank -> read.
        _progress("Retrieving relevant sources...")
        t0 = time.perf_counter()
        use_multi_hop = (
            allow_llm_strategies and self._hop_controller and self._config.enable_multi_hop
        )
        with self._retrieval_strategy_scope(allow_llm_strategies):
            if use_multi_hop:
                read_results = self._hop_controller.execute(plan.retrieval_query, profile)
            else:
                read_results = self._retrieval_pass.run(
                    plan.retrieval_query,
                    profile,
                    rewrite_result=plan.rewrite_result,
                    progress=progress,
                )
        addresses = [r.address for r in read_results]
        timings.append(("Retrieval", time.perf_counter() - t0))

        if not read_results:
            return _RetrievalOutcome(
                sanitized=sanitized, expanded=[], addresses=[], timings=timings
            )

        # 4. Expand with context
        t0 = time.perf_counter()
        expanded = self._expander.expand(
            read_results, entity_expansion_limit=profile.entity_expansion_limit
        )
        timings.append(("Expand context", time.perf_counter() - t0))

        # 4.5. Execute table queries (SQL generation + execution)
        if execute_table_queries:
            expanded = self._table_handler.process(sanitized, expanded)

        return _RetrievalOutcome(
            sanitized=sanitized, expanded=expanded, addresses=addresses, timings=timings
        )

    def _build_gap_context(
        self,
        query: str,
        governance_reasons: tuple[str, ...] = (),
    ) -> dict:
        """
        Build gap analysis context for actionable ABSTAIN messages.

        Assembles information about what the corpus DOES contain
        so the ABSTAIN message can explain gaps and suggest additions.

        Args:
            query: The user's query text
            governance_reasons: Reasons from governance constraints

        Returns:
            Dict with related_topics, top_corpus_topics, governance_reasons,
            and corpus_document_count
        """
        gap: dict = {"governance_reasons": governance_reasons}

        if not self._entity_graph_store:
            return gap

        try:
            # Extract meaningful terms from query (skip stopwords, short terms)
            _stop = {
                "what",
                "how",
                "does",
                "the",
                "this",
                "that",
                "with",
                "from",
                "have",
                "has",
                "are",
                "was",
                "were",
                "been",
                "being",
                "will",
                "would",
                "could",
                "should",
                "about",
                "which",
                "where",
                "when",
                "there",
                "their",
                "they",
                "them",
                "than",
                "then",
                "into",
                "also",
                "just",
                "only",
                "very",
                "some",
                "more",
                "most",
                "each",
                "other",
                "your",
                "our",
                "can",
                "not",
                "for",
                "and",
                "but",
            }
            terms = [t for t in query.lower().split() if len(t) > 2 and t not in _stop]

            # Find related topics via entity graph substring match
            if terms:
                gap["related_topics"] = self._entity_graph_store.find_related_topics(terms, limit=5)

            # Always include top corpus topics for context
            stats = self._entity_graph_store.stats()
            gap["top_corpus_topics"] = stats.get("top_entities", [])[:5]
            gap["corpus_document_count"] = stats.get("entities", 0)

        except Exception as e:
            logger.debug(f"Gap analysis failed: {e}")

        return gap

    def point(
        self,
        source: Path,
        collection: str | None = None,
        *,
        start_worker: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> Any:
        """Register source directory for progressive querying.

        1. Build manifest (fast, no LLM; parses + caches rich docs)
        2. Persist source_dir so future processes can find it
        3. Create AgenticSearchStrategy, wire into router
        4. Set source_dir on ContentReader (disk fallback)
        5. Optionally start BackgroundIngestWorker
        6. Return manifest immediately

        Args:
            source: Path to source directory or file
            collection: Collection name override
            start_worker: Whether to start background indexing (False for CLI)
            progress: Optional callback for status updates

        Returns:
            FileManifest with registered files
        """
        from fitz_sage.core.paths import FitzPaths
        from fitz_sage.engines.fitz_krag.progressive.builder import ManifestBuilder
        from fitz_sage.engines.fitz_krag.retrieval.strategies.agentic_search import (
            AgenticSearchStrategy,
        )

        col = collection or self._config.collection
        source = Path(source).resolve()

        if col != self._config.collection:
            self.load(col)

        # When source is a single file, use its parent as the source directory
        source_dir = source.parent if source.is_file() else source

        # 0. Stop existing background worker if re-pointing
        if self._bg_worker:
            self._bg_worker.stop()
            self._bg_worker = None

        # 1. Build manifest
        col_dir = FitzPaths.workspace() / "collections" / col
        manifest_path = col_dir / "manifest.json"
        builder = ManifestBuilder(self._config)
        manifest = builder.build(source, manifest_path, progress=progress)
        self._manifest = manifest
        self._source_dir = source_dir
        if manifest.entries():
            self._ensure_standard_llm_available(progress)

        # 2. Persist source_dir so `fitz query` can find it across processes
        col_dir.mkdir(parents=True, exist_ok=True)
        (col_dir / "source_dir.txt").write_text(str(source_dir), encoding="utf-8")

        # 3. Create agentic strategy and wire into router
        agentic = AgenticSearchStrategy(
            manifest=manifest,
            source_dir=source_dir,
            chat_factory=self._chat_factory,
            config=self._config,
            cache_dir=col_dir / "parsed",
        )
        self._retrieval_router._agentic_strategy = agentic
        self._retrieval_router._allow_llm_agentic = self._chat_factory is not None

        # 4. Set source_dir on ContentReader for disk fallback
        self._reader._source_dir = source_dir

        # 4.5. Build the ingestion core — the single parse/summarize/enrich
        # implementation shared by the synchronous bootstrap below and the
        # background worker. Bound to the engine's collection so it writes
        # the same stores retrieval reads.
        from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

        core = KragIngestPipeline(
            config=self._config,
            chat=self._chat,
            connection_manager=self._connection_manager,
            collection=self._config.collection,
            table_store=self._table_store,
            sqlite_table_store=self._sqlite_table_store,
            entity_graph_store=self._entity_graph_store,
            enricher_chat=self._enricher_chat,
            summarizer_chat=self._summarizer_chat,
        )

        # Fast synchronous symbol indexing (AST only, no LLM) via the core's
        # parse op. Populates symbol_store + import_store so LLM code search
        # works on the first query. The background worker skips these files
        # (already PARSED) and starts with summaries.
        self._fast_index_code_files(manifest, source_dir, core, progress)

        # 5. Start background worker (skip for short-lived CLI processes)
        if start_worker:
            from fitz_sage.engines.fitz_krag.progressive.worker import BackgroundIngestWorker

            self._bg_worker = BackgroundIngestWorker(
                manifest=manifest,
                source_dir=source_dir,
                core=core,
            )
            self._bg_worker.start()

        return manifest

    def _ensure_standard_llm_available(
        self,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Download and validate the managed Qwen runtime before enrichment starts."""
        ensure_available = getattr(self._enricher_chat, "ensure_available", None)
        if not callable(ensure_available):
            return

        _progress = progress or (lambda _: None)
        _progress("Preparing managed Qwen3.5 0.8B ONNX enrichment model...")
        info = ensure_available()
        revision = getattr(info, "revision", "")
        short_revision = revision[:12] if revision else "unknown"
        _progress(f"Managed Qwen ready ({short_revision}).")

    def _fast_index_code_files(
        self,
        manifest: Any,
        source_dir: Path,
        core: Any,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Fast synchronous AST symbol extraction for code files.

        Calls the ingestion core's ``parse_file`` op for each code file so
        LLM and hybrid code search work on the very first query — symbol
        signatures are indexed before LLM summaries exist.

        Files transition to PARSED so the background worker skips them and
        starts with the summarize phase.
        """
        from fitz_sage.engines.fitz_krag.ingestion.pipeline import EXTENSION_MAP
        from fitz_sage.engines.fitz_krag.progressive.manifest import FileState

        _progress = progress or (lambda _: None)

        entries = manifest.entries()
        code_entries = [e for e in entries.values() if e.file_type in EXTENSION_MAP]
        if not code_entries:
            return

        _progress(f"Indexing {len(code_entries)} code files...")

        indexed = 0
        for entry in code_entries:
            try:
                path = Path(entry.abs_path)
                if not path.exists():
                    path = source_dir / entry.rel_path
                if not path.exists():
                    continue

                core.parse_file(entry.rel_path, path, entry.file_id)
                manifest.update_state(entry.rel_path, FileState.PARSED)
                indexed += 1

            except Exception as e:
                logger.debug(f"Fast index skipped {entry.rel_path}: {e}")

        # Resolve import graph targets now that all code files are stored
        if indexed > 0:
            core.resolve_imports()
            manifest.save()
            _progress(f"Indexed {indexed} code files")

    def indexing_status(self) -> dict[str, Any]:
        """Report background-indexing progress for the loaded collection.

        Reads the disk-persisted manifest, so it reflects progress made by the
        background worker (shared across engine instances via disk).
        """
        from fitz_sage.engines.fitz_krag.progressive.manifest import (
            indexing_status as _manifest_status,
        )

        return _manifest_status(self._manifest)

    def wait_for_indexing(self, progress: Callable[[str], None] | None = None) -> None:
        """Block until background indexing finishes.

        No-op when no background worker is running (e.g. querying an
        already-loaded collection). Ctrl-C pauses indexing gracefully —
        progress is persisted and resumes on the next run.
        """
        if not self._bg_worker:
            return
        try:
            self._bg_worker.wait(progress=progress)
        except KeyboardInterrupt:
            self._bg_worker.stop()
            if progress:
                progress("Indexing paused — it resumes on the next query.")

    @property
    def config(self) -> FitzKragConfig:
        """Get the engine's configuration."""
        return self._config
