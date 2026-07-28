# tests/performance/test_latency.py
"""
Performance tests for the pipeline harness — NOT LLM speed.

Tests here measure overhead added by fitz-sage's pipeline around LLM calls:
routing, retrieval, context assembly, memory stability. They mock LLM calls
to isolate harness performance from hardware-dependent generation speed.

Run with: pytest tests/performance/ -v -s --tb=short
"""

from __future__ import annotations

import time

import pytest

from fitz_sage.core import Query

pytestmark = pytest.mark.performance


class TestHarnessOverhead:
    """Measure pipeline overhead excluding LLM generation time."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_retrieval_without_llm(self):
        """Pure retrieval (BM25 + keyword) should complete in < 5s.

        Disables LLM-backed multi-query expansion on the router to measure
        only the retrieval harness.
        """
        from dataclasses import replace

        from fitz_sage.engines.fitz_krag.retrieval_profile import build_retrieval_profile

        engine = self.runner.engine
        router = engine._retrieval_router
        base_profile = build_retrieval_profile(None, None, engine._config)
        profile = replace(base_profile, run_multi_query=False)

        saved_chat = getattr(router, "_chat_factory", None)
        router._chat_factory = None

        try:
            times = []
            for _ in range(5):
                start = time.perf_counter()
                router.retrieve("TechCorp electric vehicles", profile)
                times.append((time.perf_counter() - start) * 1000)
        finally:
            router._chat_factory = saved_chat

        p50 = sorted(times)[len(times) // 2]
        print(f"\nRetrieval-only (no LLM): p50={p50:.0f}ms")

        assert p50 < 5000, f"Retrieval harness too slow: {p50:.0f}ms (expected <5s)"

    def test_complex_query_not_much_slower_than_simple(self):
        """Complex queries should be < 4x simple query time.

        The harness overhead (more retrieval strategies, context expansion)
        should not dominate. Most time should be in LLM calls which scale
        with output length, not harness complexity.
        """
        engine = self.runner.engine

        # Measure simple query
        simple_times = []
        for _ in range(2):
            start = time.perf_counter()
            engine.answer(Query(text="Where is TechCorp headquartered?"))
            simple_times.append(time.perf_counter() - start)

        # Measure complex query
        complex_times = []
        for _ in range(2):
            start = time.perf_counter()
            engine.answer(
                Query(text="What does Sarah Chen's company's main competitor manufacture?")
            )
            complex_times.append(time.perf_counter() - start)

        simple_avg = sum(simple_times) / len(simple_times)
        complex_avg = sum(complex_times) / len(complex_times)
        ratio = complex_avg / simple_avg if simple_avg > 0 else float("inf")

        print(f"\nSimple avg: {simple_avg:.1f}s, Complex avg: {complex_avg:.1f}s")
        print(f"Ratio: {ratio:.1f}x")

        assert ratio < 4.0, (
            f"Complex query {ratio:.1f}x slower than simple — "
            f"harness overhead too high (expected <4x)"
        )


class TestMemoryUsage:
    """Memory usage benchmarks."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self, krag_e2e_runner):
        self.runner = krag_e2e_runner

    def test_query_memory_stability(self, measure_perf):
        """Memory should not grow unbounded across queries."""

        def query():
            return self.runner.engine.answer(Query(text="What is TechCorp?"))

        metrics = measure_perf(query, iterations=3, warmup=1)

        print("\nMemory Usage Across 3 Queries:")
        print(f"  Avg delta: {metrics.avg_memory_mb:.1f}MB")
        print(f"  Peak delta: {metrics.peak_memory_mb:.1f}MB")

        assert (
            metrics.peak_memory_mb < 50
        ), f"Memory grew {metrics.peak_memory_mb:.1f}MB - potential leak"
