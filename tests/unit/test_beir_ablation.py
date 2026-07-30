"""Tests for paired BEIR component-ablation reporting."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from benchmarks.fitz_bench.beir_ablation import (
    _shared_dataset_names,
    _variant_command,
    paired_delta,
)


def test_paired_delta_is_deterministic_and_reports_positive_interval() -> None:
    before = [0.0, 0.1, 0.2, 0.3, 0.4]
    after = [0.2, 0.3, 0.4, 0.5, 0.6]

    first = paired_delta(before, after, bootstrap_samples=500, seed=17)
    second = paired_delta(before, after, bootstrap_samples=500, seed=17)

    assert first == second
    assert first["mean_delta"] == pytest.approx(0.2)
    assert first["ci95_low"] > 0.0
    assert first["direction"] == "positive"


def test_paired_delta_rejects_unaligned_samples() -> None:
    with pytest.raises(ValueError, match="same positive length"):
        paired_delta([1.0], [1.0, 2.0], bootstrap_samples=100, seed=1)


def test_variant_command_omits_checkpoint_resume_when_disabled() -> None:
    args = Namespace(
        cache_dir=Path("cache"),
        workspace_root=Path("workspace"),
        index_mode="source",
        max_download_gib=2.0,
        max_extracted_gib=4.0,
        resume_queries=False,
        offline=True,
        query_limit=None,
        governance=None,
        datasets=None,
        cutoffs=[1, 10],
    )

    command = _variant_command(
        args,
        variant="literal",
        output=Path("literal.json"),
        markdown=Path("literal.md"),
    )

    assert "--resume-queries" not in command


def test_shared_dataset_names_returns_only_the_ordered_intersection() -> None:
    reports = {
        "literal": {
            "datasets": [
                {"dataset": {"name": "nfcorpus"}},
                {"dataset": {"name": "fiqa"}},
            ]
        },
        "full": {"datasets": [{"dataset": {"name": "fiqa"}}]},
    }

    assert _shared_dataset_names(reports) == ["fiqa"]
