# tests/load/test_scalability.py
"""
Scalability tests — stability under sequential load.

Verifies the harness doesn't degrade across repeated queries.
Does NOT test concurrency (local ollama serializes LLM calls).

Run with: pytest tests/load/test_scalability.py -v -s --tb=short -m scalability
"""

from __future__ import annotations

import statistics
import time

import pytest

from fitz_sage.engines.fitz_krag.retrieval_profile import build_retrieval_profile

pytestmark = pytest.mark.scalability


class TestSequentialStability:
    """Test that repeated queries don't degrade."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner
        self.router = self.runner.engine._retrieval_router
        configured_code_strategy = self.router._code_strategy
        self.router._code_strategy = getattr(
            configured_code_strategy,
            "_fallback",
            configured_code_strategy,
        )
        self.profile = build_retrieval_profile(
            None,
            None,
            self.runner.engine._config,
            keywords=[],
        )
        yield
        self.router._code_strategy = configured_code_strategy

    def test_no_throughput_degradation(self):
        """Later queries should not be significantly slower than early queries.

        Catches resource leaks, connection pool exhaustion, memory pressure.
        """
        query = "Where is TechCorp headquartered?"
        times = []

        for _ in range(10):
            start = time.perf_counter()
            self.router.retrieve(query, self.profile)
            times.append(time.perf_counter() - start)

        first = statistics.median(times[:3])
        last = statistics.median(times[-3:])
        ratio = last / first if first > 0 else float("inf")
        allowed_last = max(first * 2.0, first + 0.1)

        print("\nSequential Consistency (10 queries):")
        for i, t in enumerate(times):
            print(f"  Query {i + 1}: {t:.3f}s")
        print(f"  Median last-window/first-window: {ratio:.2f}x")

        assert last < allowed_last, (
            f"Throughput degraded: last-window median is {ratio:.1f}x the "
            f"first-window median ({last:.3f}s vs {first:.3f}s)"
        )
