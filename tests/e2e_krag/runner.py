# tests/e2e_krag/runner.py
"""
KRAG E2E test runner with ingestion and cleanup.

Equivalent to tests/e2e/runner.py but uses FitzKragEngine instead of RAGPipeline.

Handles the full lifecycle:
1. Create a unique test collection
2. Create FitzKragEngine with collection
3. Ingest test fixtures via KragIngestPipeline
4. Run test scenarios via engine.answer()
5. Clean up PostgreSQL tables for collection
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fitz_sage.logging.logger import get_logger
from tests.e2e_krag.cache import ResponseCache
from tests.e2e_krag.config import get_cache_config, get_tier_config, get_tier_names, load_e2e_config
from tests.e2e_krag.scenarios import SCENARIOS, TestScenario

from .validators import ValidationResult, validate_answer

logger = get_logger(__name__)

# Path to test fixtures (shared with RAG e2e)
FIXTURES_DIR = Path(__file__).parent / "fixtures_rag"


@dataclass
class ScenarioResult:
    """Result of running a single scenario."""

    scenario: TestScenario
    validation: ValidationResult
    answer_text: str
    duration_ms: float
    error: Optional[str] = None
    source_ids: list[str] = field(default_factory=list)


@dataclass
class E2ERunResult:
    """Result of running all E2E tests."""

    collection: str
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    ingestion_duration_s: float = 0.0
    total_duration_s: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.scenario_results if r.validation.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.scenario_results if not r.validation.passed)

    @property
    def total(self) -> int:
        return len(self.scenario_results)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


@dataclass
class TieredRunResult:
    """Result of tiered test execution."""

    results: dict[str, tuple[ScenarioResult, str]] = field(default_factory=dict)
    tier_names: list[str] = field(default_factory=list)
    cache_stats: dict = field(default_factory=dict)
    total_duration_s: float = 0.0

    def tier_summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _result, tier in self.results.values():
            counts[tier] = counts.get(tier, 0) + 1
        return counts

    @property
    def total_passed(self) -> int:
        return sum(1 for r, _t in self.results.values() if r.validation.passed)

    @property
    def total_failed(self) -> int:
        return sum(1 for r, _t in self.results.values() if not r.validation.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        return (self.total_passed / self.total * 100) if self.total > 0 else 0.0

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("TIERED KRAG E2E TEST RESULTS")
        print("=" * 60)

        tier_counts = self.tier_summary()
        cached_count = tier_counts.get("cached", 0)
        failed_all_count = tier_counts.get("failed_all", 0)

        if cached_count > 0:
            print(f"  Cached (skipped LLM):  {cached_count}")

        for tier in self.tier_names:
            count = tier_counts.get(tier, 0)
            print(f"  {tier.capitalize()} tier passed:   {count}")

        if failed_all_count > 0:
            tiers_tried = " -> ".join(self.tier_names) if self.tier_names else "none"
            print(f"  Failed ALL tiers:      {failed_all_count}  (tried: {tiers_tried})")

        print("-" * 60)
        print(f"  Total: {self.total_passed}/{self.total} passed ({self.pass_rate:.1f}%)")
        print(f"  Duration: {self.total_duration_s:.1f}s")

        if self.cache_stats:
            print(
                f"  Cache: {self.cache_stats.get('hits', 0)} hits, "
                f"{self.cache_stats.get('misses', 0)} misses"
            )
        print("=" * 60)


class KragE2ERunner:
    """
    End-to-end test runner for FitzKragEngine.

    Uses KragIngestPipeline for document ingestion and
    FitzKragEngine.answer() for query execution.

    Usage:
        runner = KragE2ERunner()
        runner.setup()  # Creates engine and ingests fixtures
        results = runner.run_all()  # Runs all scenarios
        runner.teardown()  # Cleans up collection
    """

    def __init__(
        self,
        fixtures_dir: Path | None = None,
        collection_prefix: str = "e2e_krag",
        use_cache: bool = True,
    ):
        self.fixtures_dir = fixtures_dir or FIXTURES_DIR
        self._base_collection = f"{collection_prefix}_{uuid.uuid4().hex[:8]}"
        self.engine = None
        self._setup_complete = False
        self._current_tier: str | None = None
        self._tiered_results: TieredRunResult | None = None
        self._tier_collections: set[str] = set()
        self._ingested_collections: set[str] = set()

        cache_config = get_cache_config()
        self.cache = ResponseCache(
            max_entries=cache_config.get("max_entries", 1000),
            ttl_days=cache_config.get("ttl_days", 30),
            enabled=use_cache and cache_config.get("enabled", True),
        )

    def get_tiered_result(self, scenario_id: str) -> tuple[ScenarioResult, str] | None:
        if self._tiered_results is None:
            return None
        return self._tiered_results.results.get(scenario_id)

    def setup(self, tier_name: str | None = None) -> float:
        """
        Set up the test environment.

        Creates FitzKragEngine, ingests fixtures.

        Args:
            tier_name: Which tier to initialize with (default: first tier)

        Returns:
            Duration of ingestion in seconds
        """
        from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
        from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

        logger.info(f"KRAG E2E Setup: Creating collection '{self._base_collection}'")

        start_time = time.time()

        # Load test config
        e2e_config = load_e2e_config()
        tier_names = get_tier_names(e2e_config)
        tier_name = tier_name or tier_names[0]

        logger.info(f"KRAG E2E Setup: Using test_config.yaml (tier: {tier_name})")

        tier_config = get_tier_config(tier_name, e2e_config)

        chat_plugin = tier_config["chat"]["plugin_name"]
        chat_models = tier_config["chat"].get("models", {})
        chat_base_url = tier_config["chat"].get("base_url")
        chat_api_key_env = tier_config["chat"].get("api_key_env")

        # Build flat chat specs
        chat_fast = (
            f"{chat_plugin}/{chat_models['fast']}" if chat_models.get("fast") else chat_plugin
        )
        chat_balanced = (
            f"{chat_plugin}/{chat_models['balanced']}"
            if chat_models.get("balanced")
            else chat_plugin
        )
        chat_smart = (
            f"{chat_plugin}/{chat_models['smart']}" if chat_models.get("smart") else chat_plugin
        )

        logger.info(f"KRAG E2E Setup: chat={chat_plugin} base_url={chat_base_url}")

        tier_collection = self._collection_for_tier(tier_name)
        self._tier_collections.add(tier_collection)

        # Build KRAG config — post-v0.12.0 schema (no embedding / vector_db; BM25 + KRAG routing + LLM rerank)
        config_dict = {
            "chat_fast": chat_fast,
            "chat_balanced": chat_balanced,
            "chat_smart": chat_smart,
            "collection": tier_collection,
            # Disable governance for E2E tests to isolate retrieval testing
            "governance": None,
            # Relax strict grounding for test flexibility
            "strict_grounding": False,
            # Higher retrieval for better recall
            "top_addresses": 20,
            "top_read": 10,
        }
        if chat_base_url:
            config_dict["chat_base_url"] = chat_base_url
        if chat_api_key_env:
            config_dict["chat_api_key_env"] = chat_api_key_env

        cfg = FitzKragConfig(**config_dict)
        self.engine = FitzKragEngine(cfg)

        # Ingest fixtures via KragIngestPipeline (engine no longer exposes ingest())
        logger.info(f"KRAG E2E Setup: Ingesting fixtures from '{self.fixtures_dir}'")

        stats = self._run_ingest_pipeline(self.engine, self.fixtures_dir)
        self._ingested_collections.add(tier_collection)

        ingestion_duration = time.time() - start_time
        logger.info(
            f"KRAG E2E Setup: Ingested {stats.get('files_new', 0)} files, "
            f"{stats.get('symbols_extracted', 0)} symbols, "
            f"{stats.get('sections_extracted', 0)} sections in {ingestion_duration:.1f}s"
        )

        self._setup_complete = True
        self._current_tier = tier_name
        logger.info(f"KRAG E2E Setup: Complete (tier: {tier_name})")

        return ingestion_duration

    def _collection_for_tier(self, tier_name: str) -> str:
        """Return a unique collection name for the given tier."""
        return f"{self._base_collection}_{tier_name}"

    def _rebuild_engine(self, tier_name: str) -> None:
        """
        Rebuild engine with a different tier's configuration.

        Each tier gets its own collection to avoid dimension conflicts.
        Only ingests if the tier's collection hasn't been ingested yet.

        Args:
            tier_name: Name of the tier to switch to
        """
        if tier_name == self._current_tier:
            logger.debug(f"Already on tier '{tier_name}', skipping rebuild")
            return

        logger.info(f"KRAG E2E: Switching to tier '{tier_name}'")

        # Close old engine's connection pools to free pgserver connections
        if self.engine:
            self.engine._connection_manager.close_pool(self.engine._config.collection)

        from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
        from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

        e2e_config = load_e2e_config()
        tier_config = get_tier_config(tier_name, e2e_config)

        chat_plugin = tier_config["chat"]["plugin_name"]
        chat_models = tier_config["chat"].get("models", {})
        chat_base_url = tier_config["chat"].get("base_url")
        chat_api_key_env = tier_config["chat"].get("api_key_env")

        # Build flat chat specs
        chat_fast = (
            f"{chat_plugin}/{chat_models['fast']}" if chat_models.get("fast") else chat_plugin
        )
        chat_balanced = (
            f"{chat_plugin}/{chat_models['balanced']}"
            if chat_models.get("balanced")
            else chat_plugin
        )
        chat_smart = (
            f"{chat_plugin}/{chat_models['smart']}" if chat_models.get("smart") else chat_plugin
        )

        tier_collection = self._collection_for_tier(tier_name)
        self._tier_collections.add(tier_collection)

        # Rebuild engine with new tier config and tier-specific collection
        config_dict = {
            "chat_fast": chat_fast,
            "chat_balanced": chat_balanced,
            "chat_smart": chat_smart,
            "collection": tier_collection,
            "governance": None,
            "strict_grounding": False,
            "top_addresses": 20,
            "top_read": 10,
        }
        if chat_base_url:
            config_dict["chat_base_url"] = chat_base_url
        if chat_api_key_env:
            config_dict["chat_api_key_env"] = chat_api_key_env

        cfg = FitzKragConfig(**config_dict)
        self.engine = FitzKragEngine(cfg)

        if tier_collection not in self._ingested_collections:
            logger.info(f"KRAG E2E: New collection for tier '{tier_name}', ingesting...")
            self._run_ingest_pipeline(self.engine, self.fixtures_dir)
            self._ingested_collections.add(tier_collection)

        self._current_tier = tier_name
        logger.info(f"KRAG E2E: Switched to tier '{tier_name}' (chat={chat_plugin})")

    @staticmethod
    def _run_ingest_pipeline(engine: Any, fixtures_dir: Path) -> dict[str, Any]:
        """Run KragIngestPipeline using the engine's initialized components."""
        from fitz_sage.engines.fitz_krag.ingestion.pipeline import KragIngestPipeline

        pipeline = KragIngestPipeline(
            config=engine._config,
            chat=engine._chat,
            connection_manager=engine._connection_manager,
            collection=engine._config.collection,
            table_store=engine._table_store,
            pg_table_store=engine._pg_table_store,
            vocabulary_store=engine._vocabulary_store,
            entity_graph_store=engine._entity_graph_store,
        )
        return pipeline.ingest(fixtures_dir, force=True)

    def teardown(self) -> None:
        """Clean up the test environment."""
        if not self.engine:
            return

        logger.info(f"KRAG E2E Teardown: Cleaning {len(self._tier_collections)} tier collections")

        for collection in self._tier_collections:
            try:
                self._cleanup_collection(collection)
            except Exception as e:
                logger.warning(f"KRAG E2E Teardown: Cleanup error for '{collection}': {e}")

        self.engine = None
        self._setup_complete = False
        logger.info("KRAG E2E Teardown: Complete")

    def _cleanup_collection(self, collection: str | None = None) -> None:
        """Drop PostgreSQL tables and vector data for a test collection."""
        collection = collection or self._base_collection

        from fitz_sage.storage.sqlite import SqliteConnectionManager

        try:
            conn_mgr = SqliteConnectionManager.get_instance()
            # KRAG tables use "krag_" prefix within the collection's database
            table_names = [
                "krag_raw_files",
                "krag_symbol_index",
                "krag_import_graph",
                "krag_section_index",
                "krag_table_index",
            ]
            for table_name in table_names:
                try:
                    conn_mgr.execute(collection, f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
                except Exception:
                    pass
            logger.debug(f"Dropped KRAG tables for collection '{collection}'")
        except Exception as e:
            logger.debug(f"PostgreSQL cleanup failed (non-fatal): {e}")

    def run_scenario(self, scenario: TestScenario, use_cache: bool = True) -> ScenarioResult:
        """Run a single test scenario."""
        if not self._setup_complete:
            raise RuntimeError("KRAG E2E runner not set up. Call setup() first.")

        from fitz_sage.core import Query

        logger.debug(f"Running scenario {scenario.id}: {scenario.name}")

        start_time = time.time()

        # Check cache first
        if use_cache and self.cache.enabled:
            cached = self.cache.get(scenario.query, [], scenario_id=scenario.id)
            if cached and cached.get("passed"):
                logger.debug(f"Cache hit for scenario {scenario.id}")
                duration_ms = (time.time() - start_time) * 1000
                return ScenarioResult(
                    scenario=scenario,
                    validation=ValidationResult(
                        passed=True,
                        reason="Cached result",
                        details={"cached": True, "original_tier": cached.get("tier")},
                    ),
                    answer_text=cached.get("answer_text", "")[:500],
                    duration_ms=duration_ms,
                    error=None,
                )

        # Run query through KRAG engine
        error = None
        answer_text = ""
        source_ids: list[str] = []

        try:
            answer = self.engine.answer(Query(text=scenario.query))
            answer_text = answer.text
            source_ids = [p.source_id for p in answer.provenance] if answer.provenance else []
            validation = validate_answer(answer, scenario)
        except Exception as e:
            logger.error(f"Scenario {scenario.id} failed with error: {e}")
            error = str(e)
            validation = ValidationResult(
                passed=False,
                reason=f"Engine error: {e}",
                details={"error": str(e)},
            )

        duration_ms = (time.time() - start_time) * 1000

        # Cache successful results
        if use_cache and self.cache.enabled and validation.passed:
            self.cache.set(
                query=scenario.query,
                chunk_ids=source_ids,
                scenario_id=scenario.id,
                answer_text=answer_text[:500] if answer_text else "",
                passed=True,
                tier=self._current_tier or "unknown",
            )

        return ScenarioResult(
            scenario=scenario,
            validation=validation,
            answer_text=answer_text[:500] if answer_text else "",
            duration_ms=duration_ms,
            error=error,
            source_ids=source_ids,
        )

    def run_all(self, scenarios: list[TestScenario] | None = None) -> E2ERunResult:
        """Run all test scenarios."""
        scenarios = scenarios or SCENARIOS
        results: list[ScenarioResult] = []

        logger.info(f"KRAG E2E Run: Starting {len(scenarios)} scenarios")
        start_time = time.time()

        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

            status = "PASS" if result.validation.passed else "FAIL"
            logger.info(f"  [{status}] {scenario.id}: {scenario.name} ({result.duration_ms:.0f}ms)")

        total_duration = time.time() - start_time

        run_result = E2ERunResult(
            collection=self._base_collection,
            scenario_results=results,
            total_duration_s=total_duration,
        )

        logger.info(
            f"KRAG E2E Run: Complete - {run_result.passed}/{run_result.total} passed "
            f"({run_result.pass_rate:.1f}%) in {total_duration:.1f}s"
        )

        return run_result

    def run_tiered(self, scenarios: list[TestScenario] | None = None) -> TieredRunResult:
        """Run tests with tiered fallback (local -> cloud)."""
        if not self._setup_complete:
            raise RuntimeError("KRAG E2E runner not set up. Call setup() first.")

        scenarios = scenarios or SCENARIOS
        e2e_config = load_e2e_config()
        tier_names = get_tier_names(e2e_config)

        all_results: dict[str, tuple[ScenarioResult, str]] = {}
        remaining = list(scenarios)

        logger.info(f"KRAG E2E Tiered Run: Starting {len(scenarios)} scenarios")
        logger.info(f"KRAG E2E Tiered Run: Tiers = {tier_names}")
        start_time = time.time()

        # Phase 0: Check cache
        if self.cache.enabled:
            logger.info("\n--- Checking cache ---")
            still_need_run = []
            for scenario in remaining:
                cached = self.cache.get(scenario.query, [], scenario_id=scenario.id)
                if cached and cached.get("passed"):
                    result = ScenarioResult(
                        scenario=scenario,
                        validation=ValidationResult(
                            passed=True,
                            reason="Cached result",
                            details={"cached": True, "original_tier": cached.get("tier")},
                        ),
                        answer_text=cached.get("answer_text", "")[:500],
                        duration_ms=0.0,
                        error=None,
                    )
                    all_results[scenario.id] = (result, "cached")
                    logger.info(f"  [CACHE] {scenario.id}: {scenario.name}")
                else:
                    still_need_run.append(scenario)

            cached_count = len(remaining) - len(still_need_run)
            logger.info(f"Cache: {cached_count} hits, {len(still_need_run)} need to run")
            remaining = still_need_run

        last_results: dict[str, ScenarioResult] = {}

        for tier_name in tier_names:
            if not remaining:
                break

            logger.info(f"\n--- Tier '{tier_name}': {len(remaining)} scenarios ---")
            self._rebuild_engine(tier_name)

            still_failing = []
            for scenario in remaining:
                result = self.run_scenario(scenario, use_cache=False)
                last_results[scenario.id] = result

                if result.validation.passed:
                    all_results[scenario.id] = (result, tier_name)
                    status = "PASS"
                    if self.cache.enabled:
                        self.cache.set(
                            query=scenario.query,
                            chunk_ids=result.source_ids,
                            scenario_id=scenario.id,
                            answer_text=result.answer_text,
                            passed=True,
                            tier=tier_name,
                        )
                else:
                    still_failing.append(scenario)
                    status = "FAIL"

                logger.info(
                    f"  [{status}] {scenario.id}: {scenario.name} ({result.duration_ms:.0f}ms)"
                )

            passed_this_tier = len(remaining) - len(still_failing)
            logger.info(f"Tier '{tier_name}': {passed_this_tier}/{len(remaining)} passed")
            remaining = still_failing

        for scenario in remaining:
            result = last_results[scenario.id]
            all_results[scenario.id] = (result, "failed_all")

        total_duration = time.time() - start_time

        tiered_result = TieredRunResult(
            results=all_results,
            tier_names=tier_names,
            cache_stats=self.cache.stats(),
            total_duration_s=total_duration,
        )

        tiered_result.print_summary()

        return tiered_result
