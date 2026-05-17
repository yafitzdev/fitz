# tools/retrieval_eval/metrics.py
"""Deterministic retrieval metrics: recall@k and nDCG@k over graded relevance.

A *unit* is one ground-truth relevant item (a code file, a document section, a
table value). Each unit carries:

  - grade: relevance weight — 2 = critical, 1 = relevant
  - rank:  1-indexed position of the first retrieved result that matched it,
           or None when no retrieved result matched

Every function here is pure: the same units always produce the same number.
That determinism is the point — these numbers are a regression alarm, not an
LLM judge that wobbles between runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    """One ground-truth relevant item and the rank retrieval gave it."""

    grade: int
    rank: int | None


def recall_at_k(units: list[Unit], k: int, min_grade: int = 1) -> float:
    """Fraction of relevant units that appear within the top ``k`` results.

    Args:
        units: ground-truth units for one query.
        k: rank cutoff.
        min_grade: 1 counts every unit; 2 restricts to critical units only.

    Returns:
        Recall in [0, 1]. 1.0 when no unit meets ``min_grade`` (vacuously true).
    """
    pool = [u for u in units if u.grade >= min_grade]
    if not pool:
        return 1.0
    found = sum(1 for u in pool if u.rank is not None and u.rank <= k)
    return found / len(pool)


def ndcg_at_k(units: list[Unit], k: int) -> float:
    """Normalized discounted cumulative gain at rank ``k``.

    DCG scores each unit's grade, discounted by the rank at which retrieval
    first surfaced it. IDCG is the maximum achievable DCG — every relevant unit
    surfaced at rank 1. One retrieved result can satisfy several units at once,
    so that packed ideal is reachable; normalizing by it keeps nDCG in [0, 1]
    even when several units collide on a single result.

    A perfect score therefore needs every unit in the rank-1 result. For modes
    where units map to distinct results (code, section) a flawless ranking
    still scores below 1.0 — that is fine, this is a regression alarm, not a
    leaderboard.

    Returns:
        nDCG in [0, 1]. 1.0 when there are no relevant units.
    """
    pool = [u for u in units if u.grade >= 1]
    if not pool:
        return 1.0

    dcg = sum(u.grade / math.log2(u.rank + 1) for u in pool if u.rank is not None and u.rank <= k)
    idcg = sum(u.grade for u in pool)
    return dcg / idcg


def _selfcheck() -> None:
    """Hand-computed cases — run via ``python -m tools.retrieval_eval.metrics``."""
    # Every unit in the rank-1 result: nDCG is exactly 1.0.
    packed = [Unit(2, 1), Unit(1, 1)]
    assert recall_at_k(packed, 5) == 1.0
    assert math.isclose(ndcg_at_k(packed, 5), 1.0)

    missed = [Unit(2, None), Unit(1, None)]
    assert recall_at_k(missed, 5) == 0.0
    assert ndcg_at_k(missed, 5) == 0.0

    partial = [Unit(2, 1), Unit(1, 3)]
    assert recall_at_k(partial, 5) == 1.0
    assert recall_at_k(partial, 2) == 0.5
    assert recall_at_k(partial, 2, min_grade=2) == 1.0
    # dcg = 2/log2(2) + 1/log2(4) = 2.5 ; idcg = 2 + 1 = 3
    assert math.isclose(ndcg_at_k(partial, 5), 2.5 / 3)

    # Many units colliding on one result must never push nDCG above 1.0.
    collided = [Unit(2, 1), Unit(2, 1), Unit(1, 1), Unit(1, 1)]
    assert ndcg_at_k(collided, 5) <= 1.0
    assert math.isclose(ndcg_at_k(collided, 5), 1.0)

    assert recall_at_k([Unit(1, None)], 5, min_grade=2) == 1.0

    print("metrics self-check passed")


if __name__ == "__main__":
    _selfcheck()
