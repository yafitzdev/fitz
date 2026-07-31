"""Deterministic paired statistics shared by retrieval benchmark suites."""

from __future__ import annotations

import hashlib
import random
from typing import Any


def paired_delta(
    before: list[float],
    after: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Return a deterministic paired mean difference and percentile interval."""
    if len(before) != len(after) or not before:
        raise ValueError("Paired samples must have the same positive length.")
    deltas = [right - left for left, right in zip(before, after, strict=True)]
    observed = sum(deltas) / len(deltas)
    generator = random.Random(seed)
    bootstrap = sorted(
        sum(deltas[generator.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    )
    low = _percentile(bootstrap, 0.025)
    high = _percentile(bootstrap, 0.975)
    direction = "inconclusive"
    if low > 0.0:
        direction = "positive"
    elif high < 0.0:
        direction = "negative"
    return {
        "observations": len(deltas),
        "before_mean": sum(before) / len(before),
        "after_mean": sum(after) / len(after),
        "mean_delta": observed,
        "ci95_low": low,
        "ci95_high": high,
        "direction": direction,
    }


def derived_seed(base: int, *parts: str) -> int:
    """Derive a stable independent seed from a base seed and labels."""
    payload = ":".join([str(base), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    index = round((len(values) - 1) * fraction)
    return values[index]
