# tests/unit/mock_engine.py
"""
Shared builder for a fully-mocked `FitzKragEngine`.

Several unit-test files construct a `FitzKragEngine` via `__new__` and
hand-set the ~30 attributes `answer()` touches. `build_mock_engine`
holds that boilerplate in one place; a per-file `_make_engine` helper
adds only what that file needs on top (e.g. a rewrite-aware batcher).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
from fitz_sage.engines.fitz_krag.query_batcher import BatchResult
from fitz_sage.engines.fitz_krag.query_planner import DeterministicQueryPlanner
from fitz_sage.engines.fitz_krag.retrieval.retrieval_pass import RetrievalPass


def build_mock_engine(**config_overrides) -> FitzKragEngine:
    """Build a `FitzKragEngine` with every component replaced by a mock.

    Bypasses `__init__` entirely — no real imports, no stores. Detection,
    multi-hop and rewriting are disabled; the query batcher
    returns a neutral `GENERAL` analysis. Pass `**config_overrides` to
    tweak the `FitzKragConfig` (`collection` defaults to `test_collection`).
    """
    config_kwargs = {"collection": "test_collection", **config_overrides}
    config = FitzKragConfig(**config_kwargs)
    engine = FitzKragEngine.__new__(FitzKragEngine)

    engine._config = config
    engine._chat = MagicMock(name="chat")
    engine._connection_manager = MagicMock(name="connection_manager")
    engine._raw_store = MagicMock(name="raw_store")
    engine._symbol_store = MagicMock(name="symbol_store")
    engine._import_store = MagicMock(name="import_store")
    engine._section_store = MagicMock(name="section_store")
    engine._retrieval_router = MagicMock(name="retrieval_router")
    engine._reader = MagicMock(name="reader")
    engine._expander = MagicMock(name="expander")
    engine._expander.expand.side_effect = lambda results, **kwargs: results
    engine._table_handler = MagicMock(name="table_handler")
    engine._table_handler.process.side_effect = lambda q, results, **kwargs: results
    engine._assembler = MagicMock(name="assembler")
    engine._synthesizer = MagicMock(name="synthesizer")
    governance_decision = MagicMock(name="governance_decision")
    governance_decision.mode = AnswerMode.SUFFICIENT
    governance_decision.reason = "Sources support a confident answer."
    governance_decision.reasons = ("Sources support a confident answer.",)
    governance_decision.probs = (0.1, 0.1, 0.8)
    engine._governance = MagicMock(name="governance")
    engine._governance.decide.return_value = governance_decision
    engine._governance.plan_query.return_value = SimpleNamespace(
        retrieval_intents=SimpleNamespace(
            final_labels=("needs_lookup",),
            final_label="needs_lookup",
            confidence=0.9,
            probabilities={"needs_lookup": 0.9},
        ),
        evidence_kinds=SimpleNamespace(
            final_labels=("needs_text",),
            final_label="needs_text",
            confidence=0.85,
            probabilities={"needs_text": 0.85},
        ),
    )
    engine._query_rewriter = None
    engine._address_reranker = None
    engine._hop_controller = None
    engine._retrieval_pass = RetrievalPass(
        engine._retrieval_router, engine._address_reranker, engine._reader, engine._config
    )
    engine._table_store = MagicMock(name="table_store")
    engine._sqlite_table_store = MagicMock(name="sqlite_table_store")
    engine._chat_factory = None
    engine._entity_graph_store = None
    engine._bg_worker = None
    engine._manifest = None
    engine._source_dir = None
    engine._query_planner = DeterministicQueryPlanner()

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
    engine._semantic_keyword_batcher = MagicMock(name="semantic_keyword_batcher")
    engine._semantic_keyword_batcher.batch_classify.return_value = BatchResult(keywords=[])

    return engine


__all__ = ["build_mock_engine"]
