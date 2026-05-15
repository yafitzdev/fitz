# tests/e2e_krag/test_vlm_parsing.py
"""
KRAG E2E tests for VLM-powered figure parsing via the ``endpoint``
LLM provider.

Tests that DoclingVisionParser + OpenAICompatVision correctly extract
figure descriptions from PDFs, and that KRAG retrieval can answer
questions about chart/figure data.

Requires an OpenAI-compatible HTTP server with a vision model. Any of:

- ``llama-server`` (llama.cpp) with a vision-capable GGUF on port 8080.
- LM Studio / vLLM with a vision model on port 8080.
- Ollama (``/v1/`` mode) on port 11434 with a vision model pulled.

The probe target is auto-detected on common ports (8080, 8000, 1234,
11434). Override via the env vars below if needed.

Skip behavior:
- Skips the module if no OpenAI-compatible vision endpoint is reachable.
- Skips if the configured model is not present on the server.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import httpx
import pytest

from tests.e2e_krag.scenarios import Feature, TestScenario

pytestmark = [pytest.mark.e2e_krag_parser, pytest.mark.llm]

FIXTURES_PARSER_DIR = Path(__file__).parent / "fixtures_parser"

# Override-able for CI / non-default setups.
VISION_BASE_URL = os.getenv("FITZ_E2E_VISION_BASE_URL", "")
VISION_MODEL = os.getenv("FITZ_E2E_VISION_MODEL", "llava")

# Common local OpenAI-compatible ports.
_PROBE_PORTS = (8080, 8000, 1234, 11434)


def _probe_vision_endpoint() -> str | None:
    """Find a reachable OpenAI-compatible server with the configured model."""
    if VISION_BASE_URL:
        try:
            resp = httpx.get(f"{VISION_BASE_URL}/models", timeout=2.0)
            if resp.status_code == 200:
                return VISION_BASE_URL
        except Exception:
            return None
        return None

    for port in _PROBE_PORTS:
        for host in ("localhost", "127.0.0.1"):
            base_url = f"http://{host}:{port}/v1"
            try:
                resp = httpx.get(f"{base_url}/models", timeout=1.0)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                models = data.get("data") or data.get("models") or []
                ids = [str(m.get("id") or m.get("name") or "") for m in models]
                if any(VISION_MODEL in mid for mid in ids):
                    return base_url
            except Exception:
                continue
    return None


_RESOLVED_VISION_BASE_URL = _probe_vision_endpoint()

pytestmark.append(
    pytest.mark.skipif(
        _RESOLVED_VISION_BASE_URL is None,
        reason=(
            "No OpenAI-compatible vision endpoint reachable. "
            "Start a server with a vision model (e.g. "
            "`llama-server -m llava-7b.gguf --port 8080`) or set "
            "FITZ_E2E_VISION_BASE_URL / FITZ_E2E_VISION_MODEL."
        ),
    )
)


# VLM-specific figure retrieval scenarios
VLM_SCENARIOS: list[TestScenario] = [
    TestScenario(
        id="E145",
        name="Figure: chart data retrieval",
        feature=Feature.FIGURE_RETRIEVAL,
        query="What is the projected quantum computing market size in 2028?",
        must_contain_any=["42.7", "billion", "2028"],
        min_sources=1,
    ),
    TestScenario(
        id="E146",
        name="Figure: caption information",
        feature=Feature.FIGURE_RETRIEVAL,
        query="What is the CAGR for the quantum computing market shown in the figure?",
        must_contain_any=["72.3%", "72.3", "CAGR"],
        min_sources=1,
    ),
    TestScenario(
        id="E147",
        name="Figure: market growth context",
        feature=Feature.FIGURE_RETRIEVAL,
        query=(
            "According to the market analysis figure, what was the quantum "
            "computing market size in 2024?"
        ),
        must_contain_any=["4.8", "billion", "2024"],
        min_sources=1,
    ),
]


@pytest.fixture(scope="module")
def vlm_krag_engine(set_workspace):
    """
    Module-scoped KRAG engine configured with the ``endpoint`` vision
    provider for VLM parsing.

    Creates a unique collection, ingests ``fixtures_parser/`` with
    ``docling_vision`` parser pointed at whichever OpenAI-compatible
    server we detected, then yields the engine for querying.
    """
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
    from fitz_sage.storage.sqlite import SqliteConnectionManager
    from tests.e2e_krag.config import get_tier_config, get_tier_names, load_e2e_config

    collection = f"e2e_vlm_{uuid.uuid4().hex[:8]}"

    # Load base tier config for chat models + endpoint. Post-v0.12.0:
    # no embedding / vector_db — retrieval is BM25 + KRAG routing + ONNX
    # rerank, and the endpoint is carried by chat_base_url.
    e2e_config = load_e2e_config()
    tier_names = get_tier_names(e2e_config)
    tier_config = get_tier_config(tier_names[0], e2e_config)

    chat_plugin = tier_config["chat"]["plugin_name"]
    chat_models = tier_config["chat"].get("models", {})
    chat_base_url = tier_config["chat"].get("base_url")

    chat_fast = f"{chat_plugin}/{chat_models['fast']}" if chat_models.get("fast") else chat_plugin
    chat_balanced = (
        f"{chat_plugin}/{chat_models['balanced']}" if chat_models.get("balanced") else chat_plugin
    )
    chat_smart = (
        f"{chat_plugin}/{chat_models['smart']}" if chat_models.get("smart") else chat_plugin
    )

    config_dict = {
        "chat_fast": chat_fast,
        "chat_balanced": chat_balanced,
        "chat_smart": chat_smart,
        "collection": collection,
        # VLM config — endpoint provider points at the detected URL.
        "vision": f"endpoint/{VISION_MODEL}",
        "vision_base_url": _RESOLVED_VISION_BASE_URL,
        "parser": "docling_vision",
        # Relax for testing
        "governance": None,
        "strict_grounding": False,
        "top_addresses": 20,
        "top_read": 10,
    }
    if chat_base_url:
        config_dict["chat_base_url"] = chat_base_url

    cfg = FitzKragConfig(**config_dict)
    engine = FitzKragEngine(cfg)

    # Ingest fixtures with VLM-powered parsing via KragIngestPipeline
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
    stats = pipeline.ingest(FIXTURES_PARSER_DIR, force=True)
    print(
        f"\nVLM KRAG ingest: {stats.get('files_scanned', 0)} files, "
        f"{stats.get('sections_extracted', 0)} sections"
    )

    yield engine

    # Cleanup
    try:
        conn_mgr = SqliteConnectionManager.get_instance()
        for table in [
            "krag_raw_files",
            "krag_symbol_index",
            "krag_import_graph",
            "krag_section_index",
            "krag_table_index",
        ]:
            try:
                conn_mgr.execute(collection, f'DROP TABLE IF EXISTS "{table}" CASCADE')
            except Exception:
                pass
    except Exception:
        pass


@pytest.mark.parametrize(
    "scenario",
    VLM_SCENARIOS,
    ids=lambda s: f"{s.id}_{s.feature.value}",
)
def test_vlm_figure_scenario(vlm_krag_engine, scenario):
    """
    Run a VLM figure retrieval scenario through KRAG engine.

    Validates that VLM-described figure content is retrievable and
    contains expected data values.
    """
    from fitz_sage.core import Query

    from .validators import validate_answer

    start = time.time()
    answer = vlm_krag_engine.answer(Query(text=scenario.query))
    duration_ms = (time.time() - start) * 1000

    validation = validate_answer(answer, scenario)

    if not validation.passed:
        pytest.fail(
            f"\n\nVLM Scenario {scenario.id} ({scenario.name}) FAILED\n"
            f"Feature: {scenario.feature.value}\n"
            f"Query: {scenario.query}\n"
            f"Reason: {validation.reason}\n"
            f"Details: {validation.details}\n"
            f"Answer preview: {answer.text[:300] if answer.text else '(no answer)'}...\n"
            f"Duration: {duration_ms:.0f}ms"
        )
