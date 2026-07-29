"""Unit tests for bounded ONNX cross-encoder execution."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from fitz_sage.llm.providers.onnx_reranker import OnnxReranker


def _install_fake_runtime(
    reranker: OnnxReranker,
    *,
    delay: float = 0.0,
) -> tuple[list[list[str]], list[list[str]], dict[str, int]]:
    encoded_batches: list[list[str]] = []
    executed_batches: list[list[str]] = []
    concurrency = {"active": 0, "maximum": 0}
    lock = threading.Lock()

    def encode(_queries: list[str], documents: list[str], **_kwargs: Any) -> dict[str, Any]:
        encoded_batches.append(list(documents))
        return {"documents": list(documents)}

    def run(encoded: dict[str, Any]) -> np.ndarray:
        documents = list(encoded["documents"])
        with lock:
            concurrency["active"] += 1
            concurrency["maximum"] = max(
                concurrency["maximum"],
                concurrency["active"],
            )
        try:
            if delay:
                time.sleep(delay)
            executed_batches.append(documents)
            return np.asarray([[float(len(document))] for document in documents])
        finally:
            with lock:
                concurrency["active"] -= 1

    reranker._encode = encode  # type: ignore[method-assign]
    reranker._run = run  # type: ignore[method-assign]
    return encoded_batches, executed_batches, concurrency


def test_exact_duplicates_and_cached_pairs_are_not_rescored() -> None:
    reranker = OnnxReranker(batch_size=1, workers=2, cache_size=16)
    encoded, executed, _concurrency = _install_fake_runtime(reranker)
    documents = ["alpha", "alpha", "longer beta"]

    first = reranker.rerank("query", documents, top_n=3)
    second = reranker.rerank("query", documents, top_n=3)

    assert [result.index for result in first] == [2, 0, 1]
    assert [(result.index, result.score) for result in second] == [
        (2, 11.0),
        (0, 5.0),
        (1, 5.0),
    ]
    assert len(encoded) == 2
    assert len(executed) == 2
    assert reranker.last_trace == {
        "requested_document_count": 3,
        "unique_document_count": 2,
        "duplicate_document_count": 1,
        "cache_hit_count": 2,
        "scored_document_count": 0,
        "forward_pass_count": 0,
        "batch_size": 1,
        "workers": 2,
        "cache_size": 16,
    }


def test_score_cache_is_bounded_and_does_not_retain_source_text() -> None:
    reranker = OnnxReranker(batch_size=1, workers=1, cache_size=1)
    _encoded, executed, _concurrency = _install_fake_runtime(reranker)

    reranker.rerank("query", ["alpha", "beta"])
    reranker.rerank("query", ["beta", "beta"])
    reranker.rerank("query", ["alpha", "alpha"])

    assert len(executed) == 3
    assert len(reranker._score_cache) == 1
    assert all(isinstance(key, bytes) for key in reranker._score_cache)


def test_default_execution_runs_two_batch_one_forwards_concurrently() -> None:
    reranker = OnnxReranker(cache_size=0)
    encoded, executed, concurrency = _install_fake_runtime(reranker, delay=0.03)

    reranker.rerank("query", ["one", "two", "three", "four"])

    assert all(len(batch) == 1 for batch in encoded)
    assert all(len(batch) == 1 for batch in executed)
    assert concurrency["maximum"] == 2
    assert reranker.last_trace["forward_pass_count"] == 4
    assert reranker.last_trace["batch_size"] == 1
    assert reranker.last_trace["workers"] == 2


def test_two_class_heads_use_positive_minus_negative_logit() -> None:
    reranker = OnnxReranker(batch_size=2, workers=1, cache_size=0)
    reranker._encode = lambda *_args, **_kwargs: {}  # type: ignore[method-assign]
    reranker._run = lambda _encoded: np.asarray(  # type: ignore[method-assign]
        [[1.0, 4.0], [5.0, 2.0]]
    )

    results = reranker.rerank("query", ["positive", "negative"])

    assert [(result.index, result.score) for result in results] == [
        (0, 3.0),
        (1, -3.0),
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_length": 0}, "max_length"),
        ({"batch_size": 0}, "batch_size"),
        ({"workers": 0}, "workers"),
        ({"cache_size": -1}, "cache_size"),
    ],
)
def test_invalid_execution_settings_fail_fast(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        OnnxReranker(**kwargs)
