# tests/load/locustfile.py
"""
Load testing with Locust.

Run with:
    # Start Fitz API server first (if you have one), or use direct mode
    cd tests/load
    locust -f locustfile.py --headless -u 10 -r 2 -t 60s

Options:
    -u: Number of concurrent users
    -r: Spawn rate (users per second)
    -t: Test duration
    --headless: No web UI
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from locust import User, between, task

from fitz_sage.core import Query

# Sample queries for load testing
QUERIES = [
    # Simple queries
    "Where is TechCorp headquartered?",
    "What is the price of Model Y200?",
    "Who is the CEO of TechCorp?",
    # Medium complexity
    "Compare the Model X100 vs Model Y200",
    "What employees work in Engineering?",
    "How does the authentication system work?",
    # Complex queries
    "What does Sarah Chen's company's main competitor manufacture?",
    "What security measures protect both user authentication and payment processing?",
    "List all the TechCorp vehicle models with their prices and ranges",
]


class DirectRAGUser(User):
    """
    Load test user that queries RAG pipeline directly.

    Use this when testing the library directly (no HTTP server).
    """

    wait_time = between(1, 3)  # Wait 1-3 seconds between queries
    _pipeline = None

    def on_start(self):
        """Initialize pipeline once per user."""
        if DirectRAGUser._pipeline is None:
            from fitz_sage.core.paths import FitzPaths
            from fitz_sage.engines.fitz_krag.config import FitzKragConfig
            from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

            # Set workspace
            project_root = Path(__file__).parent.parent.parent
            FitzPaths.set_workspace(project_root / ".fitz")

            # Load test config (same as all other tests)
            from tests.conftest import load_test_config

            test_config = load_test_config()

            # Use first tier (local) for load testing
            tier = test_config["tiers"][0]
            chat_plugin = tier["chat"]
            chat_models = tier.get("chat_models", {})
            config_dict = {
                "chat_fast": (
                    f"{chat_plugin}/{chat_models['fast']}"
                    if chat_models.get("fast")
                    else chat_plugin
                ),
                "chat_balanced": (
                    f"{chat_plugin}/{chat_models['balanced']}"
                    if chat_models.get("balanced")
                    else chat_plugin
                ),
                "chat_smart": (
                    f"{chat_plugin}/{chat_models['smart']}"
                    if chat_models.get("smart")
                    else chat_plugin
                ),
                "collection": "e2e_test_collection",
                "governance": "pyrrho",
                "top_addresses": 20,
            }

            cfg = FitzKragConfig(**config_dict)
            DirectRAGUser._pipeline = FitzKragEngine(cfg)

        self.pipeline = DirectRAGUser._pipeline

    @task(10)
    def simple_query(self):
        """Simple factual query (most common)."""
        query = random.choice(QUERIES[:3])
        start = time.perf_counter()

        try:
            result = self.pipeline.answer(Query(text=query))
            elapsed_ms = (time.perf_counter() - start) * 1000

            self.environment.events.request.fire(
                request_type="RAG",
                name="simple_query",
                response_time=elapsed_ms,
                response_length=len(result.text) if result else 0,
                exception=None,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.environment.events.request.fire(
                request_type="RAG",
                name="simple_query",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )

    @task(5)
    def medium_query(self):
        """Medium complexity query."""
        query = random.choice(QUERIES[3:6])
        start = time.perf_counter()

        try:
            result = self.pipeline.answer(Query(text=query))
            elapsed_ms = (time.perf_counter() - start) * 1000

            self.environment.events.request.fire(
                request_type="RAG",
                name="medium_query",
                response_time=elapsed_ms,
                response_length=len(result.text) if result else 0,
                exception=None,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.environment.events.request.fire(
                request_type="RAG",
                name="medium_query",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )

    @task(2)
    def complex_query(self):
        """Complex multi-hop query (less common)."""
        query = random.choice(QUERIES[6:])
        start = time.perf_counter()

        try:
            result = self.pipeline.answer(Query(text=query))
            elapsed_ms = (time.perf_counter() - start) * 1000

            self.environment.events.request.fire(
                request_type="RAG",
                name="complex_query",
                response_time=elapsed_ms,
                response_length=len(result.text) if result else 0,
                exception=None,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.environment.events.request.fire(
                request_type="RAG",
                name="complex_query",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
            )
