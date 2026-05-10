# fitz_sage/engines/fitz_krag/engine.py
"""
FitzKragEngine - Knowledge Routing Augmented Generation engine.

Uses knowledge-type-aware access strategies (code symbols, document sections)
instead of uniform chunk-based retrieval. Retrieval returns addresses (pointers),
content is read on demand after ranking.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fitz_sage.core import (
    Answer,
    ConfigurationError,
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

logger = get_logger(__name__)

# Fitz Cloud optimizer version for cache key computation
CLOUD_OPTIMIZER_VERSION = "1.0"


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
) -> dict[str, Any] | None:
    """
    Build a config dict for ``get_chat`` / ``get_embedder`` / ``get_vision``.

    Only includes ``base_url`` and ``auth.api_key_env`` when the provider
    spec actually consumes them. The ``openai`` preset has its own
    default URL — overriding it from the engine schema's local-default
    base_url would silently route cloud calls to localhost. So for
    non-endpoint specs we omit base_url unless the user explicitly
    overrides it.
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
    if api_key_env is not None:
        cfg["auth"] = {"api_key_env": api_key_env}

    return cfg or None


class FitzKragEngine:
    """
    Fitz KRAG engine implementation.

    Flow:
    1. Analyze query intent (+ optional detection)
    2. Retrieve addresses (pointers to code symbols / document sections)
    3. (Optional) Check cloud cache — early return on hit
    4. Read content for top-ranked addresses
    5. Expand with context (imports, class context, same-file refs)
    6. Run epistemic guardrails — determine AnswerMode
    7. Assemble LLM context
    8. Generate grounded answer with file:line provenance
    9. (Optional) Store in cloud cache
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
                    "Cannot connect to Ollama. Start it with: ollama serve\n"
                    "Config: .fitz/config.yaml"
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
        import threading
        import time as _t
        from concurrent.futures import ThreadPoolExecutor

        from fitz_sage.llm.client import get_chat, get_embedder
        from fitz_sage.storage.postgres import PostgresConnectionManager

        _t0 = _t.perf_counter()

        # LLM providers and PostgreSQL are independent — init in parallel.
        # PostgreSQL startup (pgserver) can take 1-2s; LLM provider init
        # creates HTTP clients. Overlapping saves the slower of the two.

        chat_config = _build_provider_config(
            self._config.chat_base_url,
            self._config.chat_api_key_env,
            spec=self._config.chat_smart,
        )
        embedding_config = _build_provider_config(
            self._config.embedding_base_url,
            self._config.embedding_api_key_env,
            spec=self._config.embedding,
        )

        print("  Starting database and connecting to LLM providers...", end="", flush=True)
        with ThreadPoolExecutor(max_workers=3) as pool:
            chat_future = pool.submit(
                get_chat,
                self._config.chat_smart,
                "smart",
                chat_config,
            )
            embed_future = pool.submit(
                get_embedder,
                self._config.embedding,
                embedding_config,
            )
            pg_future = pool.submit(PostgresConnectionManager.get_instance)

            self._chat = chat_future.result()
            self._embedder = embed_future.result()
            self._connection_manager = pg_future.result()
        print(" done.", flush=True)

        _t1 = _t.perf_counter()
        logger.debug(f"[init] providers+pg: {(_t1-_t0)*1000:.0f}ms")

        # Cold-start warmup: resolve embedding dimensions in background
        # so it overlaps with store creation below.
        dim_result: dict[str, int | None] = {"value": None}
        dim_error: list[Exception] = []

        def _resolve_dimensions():
            try:
                dim_result["value"] = self._embedder.dimensions
            except Exception as e:
                dim_error.append(e)

        dim_thread = threading.Thread(target=_resolve_dimensions, daemon=True)
        dim_thread.start()

        # Ingestion stores (created while embed.dimensions resolves)
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
        from fitz_sage.tabular.store.postgres import PostgresTableStore

        self._table_store = TableStore(self._connection_manager, self._config.collection)
        self._pg_table_store = PostgresTableStore(self._config.collection)

        _ts1 = _t.perf_counter()
        logger.debug(f"[init] store objects: {(_ts1-_t1)*1000:.0f}ms")

        # Retrieval (created while embed.dimensions resolves in background)
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

        code_strategy = CodeSearchStrategy(self._symbol_store, self._embedder, self._config)
        section_strategy = SectionSearchStrategy(self._section_store, self._embedder, self._config)
        table_strategy = TableSearchStrategy(self._table_store, self._embedder, self._config)
        self._retrieval_router = RetrievalRouter(
            code_strategy=code_strategy,
            chunk_strategy=None,
            config=self._config,
            section_strategy=section_strategy,
            table_strategy=table_strategy,
            chat_factory=None,  # Set after chat_factory is created
            embedder=self._embedder,
            hyde_generator=None,  # Set after HyDE generator is created
        )
        self._reader = ContentReader(
            self._raw_store,
            section_store=self._section_store,
            config=self._config,
            table_store=self._table_store,
            pg_table_store=self._pg_table_store,
        )
        self._expander = CodeExpander(
            self._raw_store,
            self._symbol_store,
            self._import_store,
            self._config,
        )

        _t3 = _t.perf_counter()
        logger.debug(f"[init] strategies: {(_t3-_ts1)*1000:.0f}ms")

        # Chat factory (shared by detection, rewriter, HyDE, multi-hop, enrichment)
        from fitz_sage.llm.factory import get_chat_factory

        # The factory uses whatever specs the user configured per tier.
        # If a user wants a single model across tiers (typical for local
        # llama-server with one model loaded), they set the same spec
        # for chat_fast / chat_balanced / chat_smart — no special-casing
        # at this layer.
        #
        # The factory currently accepts a single config dict for all
        # tiers, so per-tier base URLs aren't supported. In practice all
        # three tiers share the same chat_base_url, so this is fine.
        self._chat_factory = get_chat_factory(
            {
                "fast": self._config.chat_fast,
                "balanced": self._config.chat_balanced,
                "smart": self._config.chat_smart,
            },
            chat_config,
        )

        def _warmup_chat():
            print("  Loading LLM models (first run may take a moment)...", end="", flush=True)
            try:
                self._chat_factory("fast").chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
            except Exception:
                pass
            try:
                self._chat.chat(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=1,
                )
            except Exception:
                pass
            print(" done.", flush=True)

        threading.Thread(target=_warmup_chat, daemon=True).start()

        # Query Analysis
        from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalyzer

        self._query_analyzer = QueryAnalyzer(self._chat_factory("fast"))

        # Context + Generation
        from fitz_sage.engines.fitz_krag.context.assembler import ContextAssembler
        from fitz_sage.engines.fitz_krag.generation.synthesizer import CodeSynthesizer

        self._assembler = ContextAssembler(self._config)
        self._synthesizer = CodeSynthesizer(self._chat, self._config)

        # Guardrails (epistemic constraints)
        self._constraints: list[Any] = []
        self._governor: Any = None
        if self._config.enable_guardrails:
            from fitz_sage.governance import create_default_constraints
            from fitz_sage.governance.decider import GovernanceDecider

            self._constraints = create_default_constraints(
                chat=self._chat_factory("fast"),
                chat_balanced=self._chat_factory("balanced"),
                embedder=self._embedder,
            )
            # LLM judge disabled — requires a model capable of nuanced semantic
            # reasoning about query-evidence specificity. 7B models (command-r7b,
            # qwen2.5:7b) are too small and degrade accuracy. Enable by passing
            # chat=self._chat_factory("smart") when a capable model is available.
            self._governor = GovernanceDecider()

        # Cloud cache
        self._cloud_client: Any = None
        if self._config.cloud.get("enabled", False):
            from fitz_sage.cloud.client import CloudClient
            from fitz_sage.cloud.config import CloudConfig

            cloud_config = CloudConfig(**self._config.cloud)
            cloud_config.validate_config()
            self._cloud_client = CloudClient(cloud_config, cloud_config.org_id or "")

        # Table query handler
        from fitz_sage.engines.fitz_krag.retrieval.table_handler import TableQueryHandler

        self._table_handler = TableQueryHandler(
            self._chat_factory("balanced"), self._pg_table_store, self._config
        )

        # Shared detection
        self._detection_orchestrator: Any = None
        if self._config.enable_detection:
            from fitz_sage.retrieval.detection.registry import DetectionOrchestrator

            self._detection_orchestrator = DetectionOrchestrator(chat_factory=self._chat_factory)

        # Query rewriting
        self._query_rewriter: Any = None
        if self._config.enable_query_rewriting:
            from fitz_sage.retrieval.rewriter.rewriter import QueryRewriter

            self._query_rewriter = QueryRewriter(chat_factory=self._chat_factory)

        # Batched query intelligence (analysis + detection + rewriting in 1 LLM call)
        from fitz_sage.engines.fitz_krag.query_batcher import QueryBatcher
        from fitz_sage.retrieval.detection.modules import DEFAULT_MODULES

        self._query_batcher = QueryBatcher(
            chat_factory=self._chat_factory,
            detection_modules=list(DEFAULT_MODULES),
        )

        # HyDE generator (passed to strategies)
        self._hyde_generator: Any = None
        if self._config.enable_hyde:
            from fitz_sage.retrieval.hyde.generator import HydeGenerator

            self._hyde_generator = HydeGenerator(chat_factory=self._chat_factory)

        # Reranker (activated by config.rerank provider presence)
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

        # Wire chat_factory + HyDE into router and strategies
        if self._config.enable_multi_query:
            self._retrieval_router._chat_factory = self._chat_factory
        if self._hyde_generator:
            self._retrieval_router._hyde_generator = self._hyde_generator
        # Wire raw_store for freshness boosting
        code_strategy._raw_store = self._raw_store
        section_strategy._raw_store = self._raw_store

        # Vocabulary store + keyword matcher
        self._vocabulary_store: Any = None
        self._keyword_matcher: Any = None
        if self._config.enable_enrichment:
            try:
                from fitz_sage.retrieval.vocabulary.matcher import KeywordMatcher
                from fitz_sage.retrieval.vocabulary.store import VocabularyStore

                self._vocabulary_store = VocabularyStore(collection=self._config.collection)
                keywords = self._vocabulary_store.load()
                if keywords:
                    self._keyword_matcher = KeywordMatcher(keywords)
                    self._retrieval_router._keyword_matcher = self._keyword_matcher
            except Exception as e:
                logger.debug(f"Vocabulary store init: {e}")

        # Entity graph store
        self._entity_graph_store: Any = None
        if self._config.enable_enrichment:
            try:
                from fitz_sage.retrieval.entity_graph.store import EntityGraphStore

                self._entity_graph_store = EntityGraphStore(collection=self._config.collection)
                self._expander._entity_graph_store = self._entity_graph_store
            except Exception as e:
                logger.debug(f"Entity graph store init: {e}")

        # Multi-hop controller
        self._hop_controller: Any = None
        if self._config.enable_multi_hop:
            from fitz_sage.engines.fitz_krag.retrieval.multihop import KragHopController

            self._hop_controller = KragHopController(
                router=self._retrieval_router,
                reader=self._reader,
                chat_factory=self._chat_factory,
                max_hops=self._config.max_hops,
            )

        _t4 = _t.perf_counter()
        logger.debug(f"[init] components: {(_t4-_t3)*1000:.0f}ms")

        # Now collect embed.dimensions — should be done after ~2s of overlapped work
        dim_thread.join()
        if dim_error:
            raise dim_error[0]
        embedding_dim = dim_result["value"]
        ensure_schema(self._connection_manager, self._config.collection, embedding_dim)

        _t5 = _t.perf_counter()
        logger.debug(
            f"[init] embed.dimensions+schema: {(_t5-_t4)*1000:.0f}ms, "
            f"total: {(_t5-_t0)*1000:.0f}ms"
        )

        # Chat models already warming up from init (background threads above).

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

    def _build_detection_summary(self, results: dict, query: str) -> Any:
        """Build DetectionSummary from batched detection results.

        Adds the dict-based expansion detector (not LLM) and wraps
        module results into the same structure detect_for_retrieval returns.
        """
        from fitz_sage.retrieval.detection.protocol import (
            DetectionCategory,
            DetectionResult,
        )
        from fitz_sage.retrieval.detection.registry import DetectionSummary

        expansion_detector = self._detection_orchestrator._get_expansion_detector()
        expansion_result = expansion_detector.detect(query)

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
            expansion=expansion_result,
            rewriter=results.get(
                DetectionCategory.REWRITER,
                DetectionResult.not_detected(DetectionCategory.REWRITER),
            ),
            vocabulary=DetectionResult.not_detected(DetectionCategory.VOCABULARY),
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
              guardrails → assemble → generate → (cache store)

        Args:
            query: Query object with question text
            progress: Optional callback for status updates (e.g. ui.info)

        Returns:
            Answer with file:line provenance
        """
        import uuid

        from fitz_sage.logging import clear_query_context, set_query_context

        # Generate query ID for tracing
        query_id = f"q-{uuid.uuid4().hex[:8]}"

        # Set query context for all logging in this request
        set_query_context(query_id=query_id)
        logger.info("Starting query processing", query_length=len(query.text) if query.text else 0)

        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        if self._bg_worker:
            self._bg_worker.signal_query_start()
        try:
            # 0. Sanitize and normalize query
            import re
            import time

            sanitized = re.sub(r"<[^>]+>", "", query.text).strip()
            if not sanitized:
                sanitized = query.text.strip()

            # Truncate excessively long queries
            MAX_QUERY_LENGTH = 500
            if len(sanitized) > MAX_QUERY_LENGTH:
                original_length = len(sanitized)
                sanitized = sanitized[:MAX_QUERY_LENGTH]
                logger.debug(
                    "Query truncated", original_length=original_length, new_length=MAX_QUERY_LENGTH
                )

            _progress = progress or (lambda _: None)
            timings: list[tuple[str, float]] = []
            pipeline_start = time.perf_counter()

            # 1. Rewrite-first pipeline with parallel embed:
            #    Step 1: Rewrite query (if configured)
            #    Step 2: Batch(Analysis + Detection) on rewritten query + Embed (parallel)
            #
            # Rewriting runs first so analysis and detection classify the
            # cleaned-up query. Embed runs in parallel with classify since
            # nomic-embed-text (595MB) + chat_balanced (3.4GB) both fit in
            # VRAM — no model swap on ollama. Single chat tier eliminates
            # the old fast/balanced/smart swap problem.
            _progress("Analyzing query...")
            t0 = time.perf_counter()
            from concurrent.futures import ThreadPoolExecutor

            from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType

            retrieval_query = sanitized
            rewrite_result = None

            # Step 1: Rewrite (individual LLM call, runs first)
            if self._query_rewriter:
                try:
                    rewrite_result = self._query_rewriter.rewrite(sanitized)
                    if rewrite_result.rewritten_query != sanitized:
                        retrieval_query = rewrite_result.rewritten_query
                        logger.debug(
                            "Query rewritten",
                            original_preview=sanitized[:50],
                            rewritten_preview=retrieval_query[:50],
                        )
                except Exception as e:
                    logger.warning("Query rewriting failed, using original", error=str(e))

            # Step 2: Analysis + Detection on rewritten query, + Embedding in parallel
            classify_query = retrieval_query
            fast_analysis = self._fast_analyze(classify_query)
            need_llm_analysis = fast_analysis is None
            need_detection = bool(
                self._detection_orchestrator and self._needs_detection(classify_query)
            )

            # Run non-LLM gates to refine detection
            detection_limit_to = None
            if need_detection:
                flagged = self._detection_orchestrator.gate_categories(classify_query)
                if flagged is not None and len(flagged) == 0:
                    need_detection = False
                else:
                    detection_limit_to = flagged

            if need_llm_analysis or need_detection:
                # Batch analysis + detection in one LLM call, embed in parallel
                with ThreadPoolExecutor(max_workers=2) as pool:
                    batch_future = pool.submit(
                        self._query_batcher.batch_classify,
                        classify_query,
                        include_analysis=need_llm_analysis,
                        include_detection=need_detection,
                        detection_limit_to=detection_limit_to,
                        include_rewriting=False,
                        include_extended=True,
                    )
                    embed_future = pool.submit(self._embedder.embed_batch, [sanitized])

                    try:
                        batch_result = batch_future.result()
                        analysis = batch_result.analysis if batch_result.analysis else fast_analysis
                        detection = (
                            self._build_detection_summary(
                                batch_result.detection_results, classify_query
                            )
                            if need_detection and batch_result.detection_results is not None
                            else None
                        )
                    except Exception as e:
                        logger.warning(f"Batched query intelligence failed: {e}")
                        analysis = fast_analysis or QueryAnalysis(
                            primary_type=QueryType.GENERAL,
                            confidence=0.3,
                            refined_query=sanitized,
                        )
                        detection = None
                        batch_result = None

                    try:
                        precomputed_vectors = embed_future.result()
                        precomputed_query_vector = dict(zip([sanitized], precomputed_vectors))
                    except Exception:
                        precomputed_query_vector = None
            else:
                # No LLM needed: fast_analyze succeeded, no detection needed
                analysis = fast_analysis
                detection = None
                batch_result = None
                try:
                    precomputed_vectors = self._embedder.embed_batch([sanitized], task_type="query")
                    precomputed_query_vector = dict(zip([sanitized], precomputed_vectors))
                except Exception:
                    precomputed_query_vector = None
            timings.append(("Analysis + Detection", time.perf_counter() - t0))

            # Build retrieval profile — single object with all gates and signals
            from fitz_sage.engines.fitz_krag.retrieval_profile import build_retrieval_profile

            profile = build_retrieval_profile(
                analysis,
                detection,
                self._config,
                query_length=len(retrieval_query),
                has_chat_factory=bool(self._chat_factory),
                extended_signals=(batch_result.extended_signals if batch_result else None),
            )

            # 2. Retrieve addresses (or multi-hop)
            _progress("Retrieving relevant sources...")
            t0 = time.perf_counter()
            use_multi_hop = self._hop_controller and (
                self._config.enable_multi_hop or profile.multi_hop
            )
            if use_multi_hop:
                # Multi-hop: iterative retrieve → read → evaluate → bridge
                read_results = self._hop_controller.execute(retrieval_query, profile)
                addresses = [r.address for r in read_results] if read_results else []
            else:
                addresses = self._retrieval_router.retrieve(
                    retrieval_query,
                    profile,
                    rewrite_result=rewrite_result,
                    progress=progress,
                    precomputed_query_vectors=precomputed_query_vector,
                )
            timings.append(("Retrieval", time.perf_counter() - t0))

            if not addresses:
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

            # 2.5. Rerank addresses (when reranker configured)
            if self._address_reranker and not use_multi_hop:
                t0 = time.perf_counter()
                addresses = self._address_reranker.rerank(retrieval_query, addresses)
                timings.append(("Rerank", time.perf_counter() - t0))

            # 2.6. Check cloud cache (early return on hit)
            if self._cloud_client:
                cached = self._check_cloud_cache(query.text, addresses)
                if cached:
                    return cached

            # 3. Read content for top addresses (skip if multi-hop already read)
            if use_multi_hop:
                pass  # read_results already populated by hop controller
            else:
                _progress(
                    f"Reading content from {min(len(addresses), self._config.top_read)} sources..."
                )
                t0 = time.perf_counter()
                read_results = self._reader.read(addresses, self._config.top_read)
                timings.append(("Read content", time.perf_counter() - t0))

            if not read_results:
                _report_timings(_progress, timings, pipeline_start)
                return Answer(
                    text="Found matching symbols but could not read their content.",
                    provenance=[],
                    metadata={"engine": "fitz_krag", "query": query.text},
                )

            # 4. Expand with context
            t0 = time.perf_counter()
            expanded = self._expander.expand(
                read_results, entity_expansion_limit=profile.entity_expansion_limit
            )
            timings.append(("Expand context", time.perf_counter() - t0))

            # 4.5. Execute table queries (SQL generation + execution)
            expanded = self._table_handler.process(sanitized, expanded)

            # 5. Run guardrails (ReadResult satisfies EvidenceItem protocol)
            answer_mode = AnswerMode.TRUSTWORTHY
            if self._constraints and self._governor:
                t0 = time.perf_counter()
                from fitz_sage.governance import run_constraints
                from fitz_sage.governance.constraints.feature_extractor import extract_features

                constraint_results = run_constraints(sanitized, expanded, self._constraints)
                # Build constraint_name -> result dict for feature extraction
                cr_dict = {
                    r.metadata.get("constraint_name", f"c{i}"): r
                    for i, r in enumerate(constraint_results)
                }
                features = extract_features(
                    sanitized,
                    expanded,
                    cr_dict,
                    detection_summary=detection,
                )
                # For agentic results with no vector scores, use IE similarity as fallback
                if features.get("mean_vector_score") is None and features.get("ie_max_similarity"):
                    ie_sim = features["ie_max_similarity"]
                    features["mean_vector_score"] = ie_sim
                    features["max_vector_score"] = ie_sim
                    features["min_vector_score"] = ie_sim
                governance = self._governor.decide(
                    constraint_results, features=features, query=sanitized, chunks=expanded
                )
                answer_mode = governance.mode

                # Progressive mode override: guardrails features are degraded
                # (no summaries, no real vector scores, constraint LLMs see raw code)
                # but LLM code search already validated relevance via structural
                # reasoning. Override ABSTAIN → TRUSTWORTHY so generation runs.
                if answer_mode == AnswerMode.ABSTAIN and self._manifest is not None and expanded:
                    # Only override if abstain was solely from insufficient_evidence
                    # (expected in progressive mode due to degraded features).
                    # Preserve abstain from other constraints (real relevance failures).
                    if set(governance.triggered_constraints) <= {"insufficient_evidence"}:
                        logger.info(
                            "Overriding ABSTAIN → TRUSTWORTHY in progressive mode "
                            "(IE-only abstain with code evidence)"
                        )
                        answer_mode = AnswerMode.TRUSTWORTHY

                timings.append(("Guardrails", time.perf_counter() - t0))

            # 5.5. Compress code context (AST-based, ~50-70% token reduction)
            from fitz_sage.engines.fitz_krag.context.compressor import compress_results

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
            elif answer_mode == AnswerMode.DISPUTED and constraint_results:
                conflict_context = self._build_conflict_context(constraint_results)
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

            # 7.5. Store in cloud cache
            if self._cloud_client:
                self._store_cloud_cache(query.text, addresses, answer)

            # 7.6. Boost queried files for background worker priority
            if self._bg_worker:
                queried_paths = [
                    a.metadata.get("disk_path") for a in addresses if a.metadata.get("disk_path")
                ]
                if queried_paths:
                    self._bg_worker.boost_files(queried_paths)

            return answer

        except Exception as e:
            error_msg = str(e).lower()
            if "retriev" in error_msg or "search" in error_msg:
                raise KnowledgeError(f"Retrieval failed: {e}") from e
            elif "generat" in error_msg or "llm" in error_msg:
                raise GenerationError(f"Generation failed: {e}") from e
            else:
                raise KnowledgeError(f"KRAG pipeline error: {e}") from e
        finally:
            if self._bg_worker:
                self._bg_worker.signal_query_end()
            # Clear query context
            clear_query_context()

    def _build_conflict_context(
        self,
        constraint_results: list,
    ) -> dict | None:
        """
        Extract conflict details from ConflictAware constraint results.

        Returns dict with source names and excerpts of the conflicting
        chunks so the synthesizer can tell the LLM WHAT specifically disagrees.
        """
        for result in constraint_results:
            if result.signal != "disputed":
                continue
            meta = result.metadata
            excerpt_a = meta.get("ca_conflict_excerpt_a")
            excerpt_b = meta.get("ca_conflict_excerpt_b")
            if not excerpt_a or not excerpt_b:
                continue
            return {
                "source_a": meta.get("ca_conflict_source_a", "Source A"),
                "source_b": meta.get("ca_conflict_source_b", "Source B"),
                "excerpt_a": excerpt_a,
                "excerpt_b": excerpt_b,
                "reason": result.reason or "Sources contain conflicting information",
            }
        return None

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
            logger.debug("Gap analysis failed", error=str(e))

        return gap

    def _check_cloud_cache(self, query_text: str, addresses: list) -> Answer | None:
        """Check cloud cache for a previously cached answer."""
        from fitz_sage.cloud.cache_key import compute_retrieval_fingerprint

        try:
            query_embedding = self._embedder.embed(query_text, task_type="query")
            fingerprint = compute_retrieval_fingerprint([a.source_id for a in addresses])
            versions = self._build_cache_versions()

            result = self._cloud_client.lookup_cache(
                query_text=query_text,
                query_embedding=query_embedding,
                retrieval_fingerprint=fingerprint,
                versions=versions,
            )

            if result.hit:
                logger.info("Cloud cache hit for KRAG query")
                self._cached_query_embedding = query_embedding
                return result.answer

            self._cached_query_embedding = query_embedding
            return None
        except Exception as e:
            logger.warning(f"Cloud cache lookup failed: {e}")
            return None

    def _store_cloud_cache(self, query_text: str, addresses: list, answer: Answer) -> None:
        """Store an answer in the cloud cache."""
        from fitz_sage.cloud.cache_key import compute_retrieval_fingerprint

        try:
            query_embedding = getattr(self, "_cached_query_embedding", None)
            if query_embedding is None:
                query_embedding = self._embedder.embed(query_text, task_type="query")

            fingerprint = compute_retrieval_fingerprint([a.source_id for a in addresses])
            versions = self._build_cache_versions()

            self._cloud_client.store_cache(
                query_text=query_text,
                query_embedding=query_embedding,
                retrieval_fingerprint=fingerprint,
                versions=versions,
                answer=answer,
            )
        except Exception as e:
            logger.warning(f"Cloud cache store failed: {e}")

    def _build_cache_versions(self) -> Any:
        """Build CacheVersions for cloud cache operations."""
        import fitz_sage
        from fitz_sage.cloud.cache_key import CacheVersions

        return CacheVersions(
            optimizer=CLOUD_OPTIMIZER_VERSION,
            engine=fitz_sage.__version__,
            collection=self._config.collection,
            llm_model=self._config.chat_smart,
            prompt_template="default",
        )

    def point(
        self,
        source: Path,
        collection: str | None = None,
        *,
        start_worker: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> Any:
        """Register source directory for progressive querying.

        1. Build manifest (fast, no LLM/embedding; parses + caches rich docs)
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

        # 4. Set source_dir on ContentReader for disk fallback
        self._reader._source_dir = source_dir

        # 4.5. Fast synchronous symbol indexing (AST only, no LLM)
        # Populates symbol_store + import_store so LLM code search works
        # immediately on the first query. Background worker skips these
        # files (already PARSED) and starts with Phase 2 (summaries).
        self._fast_index_code_files(manifest, source_dir, progress)

        # 5. Start background worker (skip for short-lived CLI processes)
        if start_worker:
            from fitz_sage.engines.fitz_krag.progressive.worker import BackgroundIngestWorker

            # Create enricher for background worker if enabled
            enricher = None
            if self._config.enable_enrichment:
                from fitz_sage.engines.fitz_krag.ingestion.enricher import KragEnricher

                enricher = KragEnricher(self._chat)

            self._bg_worker = BackgroundIngestWorker(
                manifest=manifest,
                source_dir=source_dir,
                config=self._config,
                chat=self._chat,
                embedder=self._embedder,
                connection_manager=self._connection_manager,
                collection=col,
                stores={
                    "raw": self._raw_store,
                    "symbol": self._symbol_store,
                    "import": self._import_store,
                    "section": self._section_store,
                    "table": self._table_store,
                },
                vocabulary_store=self._vocabulary_store,
                entity_graph_store=self._entity_graph_store,
                pg_table_store=self._pg_table_store,
                enricher=enricher,
            )
            self._bg_worker.start()

        return manifest

    def _fast_index_code_files(
        self,
        manifest: Any,
        source_dir: Path,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Fast synchronous AST extraction + embedding for code files.

        Populates raw_store, symbol_store, and import_store so that LLM code
        search and hybrid code search work on the very first query. Embeds
        symbol signatures immediately so vector search works before LLM
        summaries are generated.

        Files are transitioned to PARSED state so the background worker skips
        Phase 1 and starts with Phase 2 (LLM summaries).
        """
        import hashlib
        import uuid

        from fitz_sage.engines.fitz_krag.ingestion.strategies.python_code import (
            PythonCodeIngestStrategy,
        )
        from fitz_sage.engines.fitz_krag.progressive.manifest import FileState

        _progress = progress or (lambda _: None)
        _CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go"}

        entries = manifest.entries()
        code_entries = [e for e in entries.values() if e.file_type in _CODE_EXTENSIONS]
        if not code_entries:
            return

        _progress(f"Indexing {len(code_entries)} code files...")

        py_strategy = PythonCodeIngestStrategy()
        ts_strategy = None
        java_strategy = None
        go_strategy = None
        indexed = 0

        for entry in code_entries:
            try:
                path = Path(entry.abs_path)
                if not path.exists():
                    path = source_dir / entry.rel_path
                if not path.exists():
                    continue

                content = path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
                content_hash = hashlib.sha256(content.encode()).hexdigest()

                # Store raw file (needed as FK for symbol_index)
                self._raw_store.upsert(
                    file_id=entry.file_id,
                    path=entry.rel_path,
                    content=content,
                    content_hash=content_hash,
                    file_type=entry.file_type,
                    size_bytes=entry.size_bytes,
                )

                # AST extraction
                result = None
                ext = entry.file_type

                if ext == ".py":
                    result = py_strategy.extract(content, entry.rel_path)
                elif ext in {".ts", ".tsx", ".js", ".jsx"}:
                    try:
                        if ts_strategy is None:
                            from fitz_sage.engines.fitz_krag.ingestion.strategies.typescript import (
                                TypeScriptIngestStrategy,
                            )

                            ts_strategy = TypeScriptIngestStrategy()
                        result = ts_strategy.extract(content, entry.rel_path)
                    except Exception:
                        pass
                elif ext == ".java":
                    try:
                        if java_strategy is None:
                            from fitz_sage.engines.fitz_krag.ingestion.strategies.java import (
                                JavaIngestStrategy,
                            )

                            java_strategy = JavaIngestStrategy()
                        result = java_strategy.extract(content, entry.rel_path)
                    except Exception:
                        pass
                elif ext == ".go":
                    try:
                        if go_strategy is None:
                            from fitz_sage.engines.fitz_krag.ingestion.strategies.go import (
                                GoIngestStrategy,
                            )

                            go_strategy = GoIngestStrategy()
                        result = go_strategy.extract(content, entry.rel_path)
                    except Exception:
                        pass

                if result is not None:
                    if result.symbols:
                        symbol_dicts = [
                            {
                                "id": str(uuid.uuid4()),
                                "name": sym.name,
                                "qualified_name": sym.qualified_name,
                                "kind": sym.kind,
                                "raw_file_id": entry.file_id,
                                "start_line": sym.start_line,
                                "end_line": sym.end_line,
                                "signature": sym.signature,
                                "summary": None,
                                "summary_vector": None,
                                "imports": sym.imports,
                                "references": sym.references,
                                "keywords": [],
                                "entities": [],
                                "metadata": {},
                            }
                            for sym in result.symbols
                        ]
                        self._symbol_store.upsert_batch(symbol_dicts)

                        # Embed signatures immediately for vector search
                        try:
                            texts = [
                                f"{s['kind']} {s['qualified_name']} {s.get('signature') or ''}"
                                for s in symbol_dicts
                            ]
                            vectors = self._embedder.embed_batch(texts, task_type="document")
                            self._symbol_store.update_vectors_by_file(entry.file_id, vectors)
                        except Exception:
                            pass  # Vector search will degrade but BM25 still works

                    if result.imports:
                        self._import_store.upsert_batch(
                            [
                                {
                                    "source_file_id": entry.file_id,
                                    "target_module": imp.target_module,
                                    "target_file_id": None,
                                    "import_names": imp.import_names,
                                }
                                for imp in result.imports
                            ]
                        )

                manifest.update_state(entry.rel_path, FileState.PARSED)
                indexed += 1

            except Exception as e:
                logger.debug(f"Fast index skipped {entry.rel_path}: {e}")

        # Resolve import graph targets
        if indexed > 0:
            try:
                path_to_id = {e.rel_path: e.file_id for e in code_entries}
                self._import_store.resolve_targets(path_to_id)
            except Exception as e:
                logger.debug(f"Import target resolution failed: {e}")

            manifest.save()
            _progress(f"Indexed {indexed} code files ({indexed} symbols ready)")

    @property
    def config(self) -> FitzKragConfig:
        """Get the engine's configuration."""
        return self._config
