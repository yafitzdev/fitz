"""Tests for KRAG retrieval-run environment fingerprints."""

from __future__ import annotations

import json

from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.run_trace import (
    _collection_sha256,
    _component_specs,
    _config_sha256,
    _strategy_executions,
)


def test_config_fingerprint_excludes_credential_configuration():
    first = FitzKragConfig(
        collection="reports",
        chat_api_key_env="FIRST_SECRET",
    )
    second = FitzKragConfig(
        collection="reports",
        chat_api_key_env="SECOND_SECRET",
    )

    assert _config_sha256(first) == _config_sha256(second)


def test_config_fingerprint_changes_with_retrieval_behavior():
    first = FitzKragConfig(collection="reports", top_addresses=20)
    second = FitzKragConfig(collection="reports", top_addresses=30)

    assert _config_sha256(first) != _config_sha256(second)


def test_default_component_specs_resolve_the_reranker_model():
    components = _component_specs(FitzKragConfig(collection="reports"))

    assert components["reranker"] == ("onnx/Alibaba-NLP/gte-reranker-modernbert-base")
    assert components["semantic_keywords"].startswith("onnx/")


def test_collection_fingerprint_tracks_manifest_and_indexing_state(tmp_path):
    manifest = tmp_path / "collections" / "reports" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"files": ["report.md"]}), encoding="utf-8")

    first = _collection_sha256(
        tmp_path,
        "reports",
        {"complete": False, "indexed": 1},
    )
    second = _collection_sha256(
        tmp_path,
        "reports",
        {"complete": True, "indexed": 1},
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_strategy_trace_includes_corpus_summary_recall():
    strategies = _strategy_executions(
        {
            "query": "summarize incidents",
            "router": {
                "strategy_calls": [
                    {
                        "strategy": "SectionSearchStrategy",
                        "query": "incidents",
                        "count": 4,
                    }
                ],
                "corpus_summary": {"enabled": True, "count": 1},
            },
        }
    )

    assert [strategy.strategy for strategy in strategies] == [
        "SectionSearchStrategy",
        "corpus_summary",
    ]
