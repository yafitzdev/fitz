"""Tests for benchmark-only retrieval component controls."""

from __future__ import annotations

from types import SimpleNamespace

from benchmarks.fitz_bench.retrieval_ablation import (
    StableTopKSelector,
    apply_ablation,
    get_ablation,
)
from fitz_sage.engines.fitz_krag.types import Address, AddressKind


def test_stable_top_k_selector_preserves_order_and_budget() -> None:
    addresses = [
        Address(
            kind=AddressKind.SECTION,
            source_id=f"source-{index}",
            location=f"section-{index}",
            summary=f"summary-{index}",
            score=float(10 - index),
        )
        for index in range(5)
    ]

    response = StableTopKSelector(k=3).rerank("query", addresses)

    assert response.addresses == addresses[:3]
    assert response.trace["reason"] == "benchmark_top_k_passthrough"
    assert response.trace["input_count"] == 5
    assert response.trace["output_count"] == 3
    assert [item["source_id"] for item in response.trace["output"]] == [
        "source-0",
        "source-1",
        "source-2",
    ]


def test_literal_ablation_disables_only_managed_components() -> None:
    semantic_batcher = object()
    canonical_reranker = object()
    query_pipeline = SimpleNamespace(_semantic_keyword_batcher=semantic_batcher)
    retrieval_pass = SimpleNamespace(_reranker=canonical_reranker)
    engine = SimpleNamespace(
        _query_pipeline=query_pipeline,
        _retrieval_pass=retrieval_pass,
        _semantic_keyword_batcher=semantic_batcher,
        _address_reranker=canonical_reranker,
        _config=SimpleNamespace(rerank_k=7),
    )

    apply_ablation(engine, get_ablation("literal"))

    assert engine._semantic_keyword_batcher is None
    assert query_pipeline._semantic_keyword_batcher is None
    assert isinstance(engine._address_reranker, StableTopKSelector)
    assert retrieval_pass._reranker is engine._address_reranker
    assert engine._address_reranker._k == 7


def test_full_ablation_keeps_canonical_components() -> None:
    semantic_batcher = object()
    canonical_reranker = object()
    query_pipeline = SimpleNamespace(_semantic_keyword_batcher=semantic_batcher)
    retrieval_pass = SimpleNamespace(_reranker=canonical_reranker)
    engine = SimpleNamespace(
        _query_pipeline=query_pipeline,
        _retrieval_pass=retrieval_pass,
        _semantic_keyword_batcher=semantic_batcher,
        _address_reranker=canonical_reranker,
        _config=SimpleNamespace(rerank_k=10),
    )

    apply_ablation(engine, get_ablation("full"))

    assert query_pipeline._semantic_keyword_batcher is semantic_batcher
    assert retrieval_pass._reranker is canonical_reranker
    assert engine._address_reranker is canonical_reranker
