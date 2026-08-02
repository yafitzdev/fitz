"""Tests for labeled retrieval metrics and failure attribution."""

from __future__ import annotations

import math

import pytest

from benchmarks.fitz_bench.retrieval_eval import (
    PlainBm25,
    aggregate_metrics,
    ranking_metrics,
    stage_failure,
    stage_recoveries,
)


def test_plain_bm25_ranks_rare_matching_document_first() -> None:
    index = PlainBm25.build(
        [
            ("common", "battery battery vehicle"),
            ("rare", "vehicle zephyrlatch"),
            ("other", "vehicle diagnostics"),
        ]
    )

    assert index.search("zephyrlatch", top_k=3) == ["rare"]


def test_ranking_metrics_use_graded_ndcg_and_binary_relevance() -> None:
    metrics = ranking_metrics(
        ["b", "x", "a"],
        {"a": 2, "b": 1, "c": 0},
        [3],
    )

    ideal_dcg = 2.0 + 1.0 / math.log2(3)
    assert metrics["Precision@3"] == pytest.approx(2 / 3)
    assert metrics["Recall@3"] == 1.0
    assert metrics["MRR@3"] == 1.0
    assert metrics["MAP@3"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert metrics["NDCG@3"] == pytest.approx(2.0 / ideal_dcg)


def test_ranking_metrics_deduplicate_document_ids() -> None:
    metrics = ranking_metrics(["a", "a", "b"], {"a": 1, "b": 1}, [2])

    assert metrics["Recall@2"] == 1.0
    assert metrics["Precision@2"] == 1.0


def test_aggregate_metrics_uses_query_macro_average() -> None:
    assert aggregate_metrics([{"Recall@5": 1.0}, {"Recall@5": 0.0}]) == {"Recall@5": 0.5}


@pytest.mark.parametrize(
    ("stages", "expected"),
    [
        ({"recall": [], "final": [], "compiled": [], "delivered": []}, "recall"),
        ({"recall": ["d"], "final": [], "compiled": [], "delivered": []}, "final"),
        (
            {"recall": ["d"], "final": ["d"], "compiled": [], "delivered": []},
            "compiled",
        ),
        (
            {"recall": ["d"], "final": ["d"], "compiled": ["d"], "delivered": []},
            "delivered",
        ),
        (
            {"recall": ["d"], "final": ["d"], "compiled": ["d"], "delivered": ["d"]},
            "delivered_hit",
        ),
    ],
)
def test_stage_failure_attributes_first_irreversible_boundary(stages, expected) -> None:
    assert stage_failure(stages, {"d": 1}) == expected


def test_stage_recoveries_reports_final_selection_rescue() -> None:
    stages = {
        "reranked": ["other"],
        "final": ["relevant"],
    }

    assert stage_recoveries(stages, {"relevant": 1}) == ["final_rescued_reranker_miss"]
