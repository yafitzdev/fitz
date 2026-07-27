# fitz_sage/engines/fitz_krag/engine.py
"""
FitzKragEngine - Knowledge Routing Augmented Generation engine.

Uses knowledge-type-aware access strategies (code symbols, document sections)
instead of uniform chunk-based retrieval. Retrieval returns addresses (pointers),
content is read on demand after ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

from fitz_sage.core import (
    Answer,
    ConfigurationError,
    EngineError,
    EvidenceItem,
    EvidencePack,
    GenerationError,
    KnowledgeError,
    Query,
    QueryError,
    RetrievalRun,
)
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.core.collections import validate_collection_name
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.evidence_compiler import compile_evidence
from fitz_sage.integrations.pyrrho import (
    answer_mode_from_pyrrho,
    decide,
    decision_payload,
)
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from pyrrho import GovernanceDecision

    from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis
    from fitz_sage.engines.fitz_krag.query_pipeline import RetrievalOutcome
    from fitz_sage.engines.fitz_krag.types import ReadResult

logger = get_logger(__name__)


@dataclass
class _GovernedEvidenceResult:
    """Internal carrier for one canonical governed retrieval execution."""

    pack: EvidencePack
    selected: list["ReadResult"]
    outcome: "RetrievalOutcome"
    compilation: Any
    decision: "GovernanceDecision"


def _evidence_delivery_limit(
    result_count: int,
    requested_limit: Any,
    *,
    default_limit: int,
) -> int:
    """Choose a fixed evidence budget without consulting a governance verdict."""
    value = default_limit if requested_limit is None else requested_limit
    if isinstance(value, bool):
        raise ValueError("top_k must be a positive integer.")
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k must be a positive integer.") from exc
    if limit <= 0:
        raise ValueError("top_k must be a positive integer.")
    return min(result_count, limit)


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
            self._initialized_collection: str | None = None
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
        collection = validate_collection_name(collection)
        if getattr(self, "_initialized_collection", None) == collection:
            if not self._manifest or not self._source_dir:
                self._try_load_persisted_manifest(collection)
            return

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

        if self._manifest is None or self._source_dir is None:
            raise RuntimeError("Agentic strategy requires an active source manifest")
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
        section_strategy = SectionSearchStrategy(
            self._section_store,
            self._raw_store,
            self._config,
        )
        table_strategy = TableSearchStrategy(
            self._table_store,
            self._config,
            self._sqlite_table_store,
        )
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

        # Chat factory for optional tiered LLM paths. Retrieval-first defaults
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

        # Pyrrho lazily loads its authoritative governance runtime on first use.
        from fitz_sage.integrations.pyrrho import create_pyrrho

        self._pyrrho = create_pyrrho(self._config.governance)

        # Table query handler
        from fitz_sage.engines.fitz_krag.retrieval.table_handler import TableQueryHandler

        table_chat = self._chat_factory("balanced") if self._chat_factory else None
        self._table_handler = TableQueryHandler(
            table_chat,
            self._sqlite_table_store,
            self._config,
        )

        # Query prep defaults to the deterministic planner. If
        # query_intelligence is configured, the batcher uses that provider and
        # treats provider/model failures as query failures.
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
        self._semantic_keyword_batcher = QueryBatcher(
            chat_factory=lambda _tier: self._enricher_chat,
            detection_modules=[],
        )

        # Reranker — mandatory INT8 ONNX cross-encoder via get_reranker().
        # Default backbone: Alibaba-NLP/gte-reranker-modernbert-base.
        # Override with rerank: "onnx/<hf-model-id>".
        from fitz_sage.llm.client import get_reranker

        reranker = get_reranker(self._config.rerank)
        if reranker is None:
            raise ConfigurationError("Reranking is mandatory; configure rerank: onnx.")

        from fitz_sage.engines.fitz_krag.retrieval.reranker import AddressReranker

        self._address_reranker = AddressReranker(
            reranker=reranker,
            k=self._config.rerank_k,
            min_addresses=self._config.rerank_min_addresses,
        )

        # LLM structural code search (default when chat available)
        active_code_strategy: Any = code_strategy
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
            active_code_strategy = llm_strategy

        # Wire raw_store for freshness boosting
        active_code_strategy._raw_store = self._raw_store
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
                pyrrho=self._pyrrho,
                max_hops=self._config.max_hops,
            )

        self._query_pipeline = self._build_query_pipeline()

        _t4 = _t.perf_counter()
        logger.debug(f"[init] components: {(_t4-_t3)*1000:.0f}ms")

        ensure_schema(self._connection_manager, self._config.collection)
        self._initialized_collection = self._config.collection

        _t5 = _t.perf_counter()
        logger.debug(f"[init] schema: {(_t5-_t4)*1000:.0f}ms, " f"total: {(_t5-_t0)*1000:.0f}ms")

        # Retrieval-first init intentionally does not create or warm chat clients.

    def _chat_tier_specs(self) -> dict[str, str] | None:
        """Return complete tier specs when optional chat tiers are configured."""
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
                from fitz_sage.engines.fitz_krag.context.compressor import compress_results

                _progress = progress or (lambda _: None)
                pipeline_start = time.perf_counter()
                try:
                    pack, selected = self._governed_evidence(query, progress=progress)
                except EngineError:
                    raise
                except Exception as e:
                    raise KnowledgeError(f"Retrieval failed: {e}") from e

                answer_mode = pack.mode
                if answer_mode is None:
                    raise KnowledgeError("Governed evidence pack is missing Pyrrho's verdict.")
                if not selected:
                    early_gap_context = self._build_gap_context(pack.query, pack.reasons)
                    _report_timings(_progress, list(pack.timings.items()), pipeline_start)
                    return Answer(
                        text=self._synthesizer._build_insufficient_message(
                            pack.query,
                            early_gap_context,
                        ),
                        provenance=[],
                        mode=answer_mode,
                        metadata={
                            "engine": "fitz_krag",
                            "query": pack.query,
                            "answer_mode": answer_mode.value,
                            "gap_context": early_gap_context,
                            "query_profile": pack.metadata.get("query_profile", {}),
                            "evidence_compiler": pack.metadata.get("evidence_compiler", {}),
                            "evidence_closure": pack.metadata.get("evidence_closure", {}),
                            "pyrrho": pack.metadata.get("pyrrho", {}),
                        },
                    )

                compressed = compress_results(selected)
                context = self._assembler.assemble(pack.query, compressed) if compressed else ""
                _progress("Generating answer...")
                t0 = time.perf_counter()
                gap_context: dict[str, Any] | None = None
                conflict_context: dict[str, Any] | None = None
                if answer_mode == AnswerMode.INSUFFICIENT:
                    gap_context = self._build_gap_context(pack.query, pack.reasons)
                elif answer_mode == AnswerMode.DISPUTED:
                    conflict_context = self._build_conflict_context(pack, selected)
                try:
                    answer = self._synthesizer.generate(
                        pack.query,
                        context,
                        compressed,
                        answer_mode=answer_mode,
                        gap_context=gap_context,
                        conflict_context=conflict_context,
                    )
                except EngineError:
                    raise
                except Exception as e:
                    raise GenerationError(f"Generation failed: {e}") from e
                answer.metadata["query_profile"] = pack.metadata.get("query_profile", {})
                answer.metadata["evidence_compiler"] = pack.metadata.get("evidence_compiler", {})
                answer.metadata["evidence_closure"] = pack.metadata.get("evidence_closure", {})
                answer.metadata["pyrrho"] = pack.metadata.get("pyrrho", {})
                timings = list(pack.timings.items())
                timings.append(("Generation", time.perf_counter() - t0))
                _report_timings(_progress, timings, pipeline_start)

                return answer

            except EngineError:
                raise
            except Exception as e:
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
            except EngineError:
                raise
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
        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        with self._query_scope():
            logger.info(f"Starting evidence retrieval (query_length={len(query.text)})")
            try:
                pack, _ = self._governed_evidence(query, progress=progress, top_k=top_k)
                return pack
            except EngineError:
                raise
            except Exception as e:
                raise KnowledgeError(f"Evidence retrieval failed: {e}") from e

    def trace(
        self,
        query: Query,
        *,
        progress: Callable[[str], None] | None = None,
        top_k: int | None = None,
    ) -> RetrievalRun:
        """Return a versioned execution record for one governed retrieval."""
        if not query.text or not query.text.strip():
            raise QueryError("Query text cannot be empty")

        with self._query_scope():
            logger.info(f"Starting traced retrieval (query_length={len(query.text)})")
            try:
                result = self._governed_result(
                    query,
                    progress=progress,
                    top_k=top_k,
                )
                from fitz_sage.core.paths import FitzPaths
                from fitz_sage.engines.fitz_krag.run_trace import (
                    build_retrieval_run,
                )

                return build_retrieval_run(
                    source_query=query.text,
                    pack=result.pack,
                    outcome=result.outcome,
                    compilation=result.compilation,
                    selected=result.selected,
                    decision=result.decision,
                    config=self._config,
                    indexing_status=result.pack.indexing_status,
                    workspace=FitzPaths.workspace(),
                )
            except EngineError:
                raise
            except Exception as e:
                raise KnowledgeError(f"Retrieval trace failed: {e}") from e

    def _governed_evidence(
        self,
        query: Query,
        *,
        progress: Callable[[str], None] | None = None,
        top_k: int | None = None,
    ) -> tuple[EvidencePack, list["ReadResult"]]:
        """Run canonical retrieval, closure, compilation, and Pyrrho governance."""
        result = self._governed_result(
            query,
            progress=progress,
            top_k=top_k,
        )
        return result.pack, result.selected

    def _governed_result(
        self,
        query: Query,
        *,
        progress: Callable[[str], None] | None = None,
        top_k: int | None = None,
    ) -> _GovernedEvidenceResult:
        """Return all products of the canonical governed retrieval path."""
        outcome = self._retrieve_core(
            query,
            progress=progress,
            use_query_intelligence=None,
            allow_llm_strategies=True,
            execute_table_queries=True,
            allow_table_sql_generation=True,
            expand_context=True,
        )
        compilation = compile_evidence(
            outcome.sanitized,
            outcome.expanded,
            profile=outcome.profile,
        )
        query_pipeline = self._build_query_pipeline()
        outcome = query_pipeline.close_evidence(
            outcome,
            compilation,
            progress=progress,
            allow_llm_strategies=True,
            execute_table_queries=True,
            allow_table_sql_generation=True,
            expand_context=True,
        )
        closure_metadata = outcome.retrieval_trace.get("evidence_closure", {})
        if closure_metadata.get("added") or closure_metadata.get("replaced"):
            compilation = compile_evidence(
                outcome.sanitized,
                outcome.expanded,
                profile=outcome.profile,
            )

        requested_top_k = top_k if top_k is not None else query.metadata.get("top_k")
        evidence_limit = _evidence_delivery_limit(
            len(compilation.results),
            requested_top_k,
            default_limit=self._config.top_read,
        )
        selected = list(compilation.results[:evidence_limit])
        import time

        pyrrho_start = time.perf_counter()
        decision = decide(
            self._pyrrho,
            outcome.sanitized,
            selected,
        )
        pyrrho_timing = ("Pyrrho", time.perf_counter() - pyrrho_start)
        mode = answer_mode_from_pyrrho(decision)
        timings = list(outcome.timings)
        timings.append(pyrrho_timing)
        self._boost_queried_files(outcome)
        pack = EvidencePack(
            query=outcome.sanitized,
            mode=mode,
            items=self._build_evidence_items(selected),
            reasons=list(decision.reasons),
            timings={name: duration for name, duration in timings},
            indexing_status=self.indexing_status(),
            metadata={
                "engine": "fitz_krag",
                "source_query": query.text,
                "query_profile": outcome.query_profile_metadata,
                "retrieval_trace": outcome.retrieval_trace,
                "evidence_closure": closure_metadata,
                "evidence_compiler": compilation.metadata,
                "pyrrho": decision_payload(decision),
                "evidence_delivery": {
                    "available": len(compilation.results),
                    "selected": len(selected),
                    "limit": evidence_limit,
                },
            },
        )
        return _GovernedEvidenceResult(
            pack=pack,
            selected=selected,
            outcome=outcome,
            compilation=compilation,
            decision=decision,
        )

    @staticmethod
    def _build_conflict_context(
        pack: EvidencePack,
        selected: list["ReadResult"],
    ) -> dict[str, str]:
        """Build concrete disputed-source context from governed evidence."""
        context = {"reason": "; ".join(pack.reasons)}
        if selected:
            context.update(
                {
                    "source_a": selected[0].file_path,
                    "excerpt_a": selected[0].content[:200],
                }
            )
        if len(selected) > 1:
            context.update(
                {
                    "source_b": selected[1].file_path,
                    "excerpt_b": selected[1].content[:200],
                }
            )
        return context

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

    def _boost_queried_files(self, outcome: "RetrievalOutcome") -> None:
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

    def _build_query_pipeline(self) -> Any:
        """Build the query-side retrieval pipeline from current engine components."""
        from fitz_sage.engines.fitz_krag.query_pipeline import QueryPipeline

        return QueryPipeline(
            config=self._config,
            query_planner=getattr(self, "_query_planner", None),
            query_batcher=self._query_batcher,
            semantic_keyword_batcher=getattr(self, "_semantic_keyword_batcher", None),
            pyrrho=self._pyrrho,
            retrieval_pass=self._retrieval_pass,
            hop_controller=self._hop_controller,
            expander=self._expander,
            table_handler=self._table_handler,
            retrieval_strategy_scope=self._retrieval_strategy_scope,
            fast_analyze=self._fast_analyze,
            needs_detection=self._needs_detection,
            build_detection_summary=self._build_detection_summary,
        )

    def _retrieve_core(
        self,
        query: Query,
        *,
        progress: Callable[[str], None] | None = None,
        use_query_intelligence: bool | None = None,
        allow_llm_strategies: bool = True,
        execute_table_queries: bool = True,
        allow_table_sql_generation: bool = True,
        expand_context: bool = True,
    ) -> "RetrievalOutcome":
        """Run the retrieval half of the KRAG pipeline.

        Analyze + detect → retrieve → read → expand → table queries. Returns
        the expanded ReadResults (pre-governance, pre-compression). Callers
        guarantee non-empty query text and supply the query-scoped context.
        """
        pipeline = getattr(self, "_query_pipeline", None) or self._build_query_pipeline()
        return cast(
            "RetrievalOutcome",
            pipeline.retrieve(
                query,
                progress=progress,
                use_query_intelligence=use_query_intelligence,
                allow_llm_strategies=allow_llm_strategies,
                execute_table_queries=execute_table_queries,
                allow_table_sql_generation=allow_table_sql_generation,
                expand_context=expand_context,
            ),
        )

    def _build_gap_context(
        self,
        query: str,
        decision_reasons: Sequence[str] = (),
    ) -> dict:
        """
        Build gap analysis context for actionable INSUFFICIENT messages.

        Assembles information about what the corpus DOES contain
        so the INSUFFICIENT message can explain gaps and suggest additions.

        Args:
            query: The user's query text
            decision_reasons: Reasons returned by Pyrrho

        Returns:
            Dict with related_topics, top_corpus_topics, decision_reasons,
            and corpus_document_count
        """
        gap: dict = {"decision_reasons": decision_reasons}

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

        col = validate_collection_name(collection or self._config.collection)
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
        manifest_entries = manifest.entries()
        self._manifest = manifest
        self._source_dir = source_dir
        if any(
            entry.state.value not in {"failed", "unsupported"}
            for entry in manifest_entries.values()
        ):
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
        core = self._build_ingest_core()
        core.delete_files_not_in_paths(set(manifest_entries))
        for entry in manifest_entries.values():
            if entry.state.value in {"failed", "unsupported"}:
                core.discard_file(entry.file_id)

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

    def continue_indexing(self) -> None:
        """Continue persisted indexing for the loaded collection, then exit."""
        if not self._manifest or not self._source_dir:
            return

        from fitz_sage.engines.fitz_krag.progressive.worker import BackgroundIngestWorker

        core = self._build_ingest_core()
        worker = BackgroundIngestWorker(
            manifest=self._manifest,
            source_dir=self._source_dir,
            core=core,
        )
        worker.run_until_deep_complete()

    def stop_background_indexing(self) -> None:
        """Stop the in-process background worker if one is running."""
        if self._bg_worker:
            self._bg_worker.stop()
            self._bg_worker = None

    def _build_ingest_core(self) -> Any:
        """Build the shared KRAG ingestion core for the current collection."""
        from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

        return KragIngestPipeline(
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

    def _ensure_standard_llm_available(
        self,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        """Download and validate the managed Qwen runtime before enrichment starts."""
        ensure_available = getattr(self._enricher_chat, "ensure_available", None)
        if not callable(ensure_available):
            return

        _progress = progress or (lambda _: None)
        _progress("Preparing managed Qwen3 0.6B ONNX GenAI enrichment snapshot...")
        info = ensure_available()
        revision = getattr(info, "revision", "")
        short_revision = revision[:12] if revision else "unknown"
        _progress(f"Managed Qwen snapshot ready ({short_revision}).")

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
        from fitz_sage.engines.fitz_krag.ingestion.formats import CODE_EXTENSION_MAP
        from fitz_sage.engines.fitz_krag.progressive.manifest import FileState

        _progress = progress or (lambda _: None)

        entries = manifest.entries()
        code_entries = [
            entry
            for entry in entries.values()
            if entry.file_type in CODE_EXTENSION_MAP and entry.state.value == "registered"
        ]
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
                core.discard_file(entry.file_id)
                manifest.mark_failed(
                    entry.rel_path,
                    stage="parse",
                    message=str(e),
                )
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
        """Block until background indexing reaches the query-ready keyword phase.

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

    def wait_for_query_surface(self, progress: Callable[[str], None] | None = None) -> None:
        """Block until parsed retrieval units are searchable."""
        if not self._bg_worker:
            return
        self._bg_worker.wait_for_query_surface(progress=progress)

    @property
    def config(self) -> FitzKragConfig:
        """Get the engine's configuration."""
        return self._config
