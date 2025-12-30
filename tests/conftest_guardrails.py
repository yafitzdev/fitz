# tests/conftest_guardrails.py
"""
Test fixtures and utilities for guardrails tests.

Provides mock embedders for testing semantic matching without
requiring actual embedding API calls.
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable

import pytest

from fitz_ai.core.guardrails import SemanticMatcher


def create_deterministic_embedder(dimension: int = 384) -> Callable[[str], list[float]]:
    """
    Create a deterministic mock embedder for testing.

    This uses a simple approach: texts that should be "similar" for our tests
    are mapped to specific vector clusters.
    """

    # Pre-defined clusters for different semantic categories
    # Each cluster is a base vector that related texts will be similar to
    def _make_cluster_vector(seed: str, cluster_id: int) -> list[float]:
        """Create a vector for a cluster."""
        h = hashlib.sha256(f"{seed}_{cluster_id}".encode()).hexdigest()
        vec = []
        for i in range(0, min(len(h), dimension * 2), 2):
            val = (int(h[i : i + 2], 16) - 128) / 128
            vec.append(val)
        while len(vec) < dimension:
            vec.append(0.0)
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec[:dimension]

    # Create base cluster vectors
    CAUSAL_QUERY_CLUSTER = _make_cluster_vector("causal_query", 1)
    CAUSAL_EVIDENCE_CLUSTER = _make_cluster_vector("causal_evidence", 2)
    FACT_QUERY_CLUSTER = _make_cluster_vector("fact_query", 3)
    RESOLUTION_QUERY_CLUSTER = _make_cluster_vector("resolution_query", 4)
    _ASSERTION_CLUSTER = _make_cluster_vector("assertion", 5)  # Reserved for future use
    # Opposition clusters
    SUCCESS_CLUSTER = _make_cluster_vector("success", 6)
    FAILURE_CLUSTER = _make_cluster_vector("failure", 7)
    SECURITY_CLUSTER = _make_cluster_vector("security", 8)
    OPERATIONAL_CLUSTER = _make_cluster_vector("operational", 9)
    IMPROVED_CLUSTER = _make_cluster_vector("improved", 10)
    DECLINED_CLUSTER = _make_cluster_vector("declined", 11)
    POSITIVE_CLUSTER = _make_cluster_vector("positive", 12)
    NEGATIVE_CLUSTER = _make_cluster_vector("negative", 13)
    NEUTRAL_CLUSTER = _make_cluster_vector("neutral", 14)

    def _add_noise(vec: list[float], text: str, noise_level: float = 0.005) -> list[float]:
        """Add text-specific noise to maintain uniqueness.

        Noise must be small enough that vectors from the same cluster
        have cosine similarity > 0.70 (the threshold used in tests).
        """
        h = hashlib.md5(text.encode()).hexdigest()
        result = []
        for i, v in enumerate(vec):
            noise = ((int(h[i % len(h)], 16) - 8) / 8) * noise_level
            result.append(v + noise)
        # Re-normalize
        norm = math.sqrt(sum(v * v for v in result))
        if norm > 0:
            result = [v / norm for v in result]
        return result

    def embed(text: str) -> list[float]:
        """
        Embed text by matching to semantic clusters.

        Returns vectors similar to the appropriate cluster based on text content.
        Priority: More specific patterns first, then general patterns.
        """
        text_lower = text.lower().strip()

        # Causal queries - "why", "what caused", "explain" (highest priority for queries)
        if text_lower.startswith("why ") or text_lower.startswith("why?"):
            return _add_noise(CAUSAL_QUERY_CLUSTER, text)
        if "what caused" in text_lower or "what led to" in text_lower:
            return _add_noise(CAUSAL_QUERY_CLUSTER, text)
        if text_lower.startswith("explain"):
            return _add_noise(CAUSAL_QUERY_CLUSTER, text)
        if "how come" in text_lower:
            return _add_noise(CAUSAL_QUERY_CLUSTER, text)

        # Resolution queries
        if "authoritative" in text_lower or "trust" in text_lower:
            return _add_noise(RESOLUTION_QUERY_CLUSTER, text)
        if "resolve" in text_lower or "reconcile" in text_lower:
            return _add_noise(RESOLUTION_QUERY_CLUSTER, text)
        if "which" in text_lower and "correct" in text_lower:
            return _add_noise(RESOLUTION_QUERY_CLUSTER, text)

        # Causal evidence markers (high priority for epistemic detection)
        causal_markers = [
            "because",
            "due to",
            "caused by",
            "led to",
            "result of",
            "therefore",
            "thus",
            "hence",
            "consequence",
            "consequently",
            "owing to",
            "attributed",
            "triggered by",
            "the reason",
            "the cause",
            "reason is",
            "reason was",
        ]
        if any(m in text_lower for m in causal_markers):
            return _add_noise(CAUSAL_EVIDENCE_CLUSTER, text)

        # Opposition: Security vs Operational (specific multi-word patterns)
        if "security incident" in text_lower or "unauthorized access" in text_lower:
            return _add_noise(SECURITY_CLUSTER, text)
        if "operational incident" in text_lower or "misconfigured" in text_lower:
            return _add_noise(OPERATIONAL_CLUSTER, text)

        # Opposition: Success vs Failure
        if any(m in text_lower for m in ["successful", "completed", "approved", "accepted"]):
            return _add_noise(SUCCESS_CLUSTER, text)
        if any(m in text_lower for m in ["failed", "rejected", "denied"]):
            return _add_noise(FAILURE_CLUSTER, text)

        # Opposition: Improved vs Declined
        if any(m in text_lower for m in ["improved", "increased", "grew"]):
            return _add_noise(IMPROVED_CLUSTER, text)
        if any(m in text_lower for m in ["declined", "decreased", "dropped"]):
            return _add_noise(DECLINED_CLUSTER, text)

        # Opposition: Positive vs Negative sentiment
        if any(m in text_lower for m in ["positive", "good", "excellent", "great"]):
            return _add_noise(POSITIVE_CLUSTER, text)
        if any(m in text_lower for m in ["negative", "bad", "poor", "terrible"]):
            return _add_noise(NEGATIVE_CLUSTER, text)

        # Fact queries
        if any(text_lower.startswith(q) for q in ["what ", "which ", "who ", "where ", "when "]):
            return _add_noise(FACT_QUERY_CLUSTER, text)

        # Default: neutral cluster (don't map assertions to ASSERTION_CLUSTER)
        # This prevents chunks like "Helios was deprecated" from matching causal concepts
        # Use slightly higher noise for neutral to ensure it doesn't accidentally match other clusters
        return _add_noise(NEUTRAL_CLUSTER, text, noise_level=0.01)

    return embed


@pytest.fixture
def mock_embedder() -> Callable[[str], list[float]]:
    """Fixture providing a deterministic mock embedder."""
    return create_deterministic_embedder()


@pytest.fixture
def semantic_matcher(mock_embedder) -> SemanticMatcher:
    """Fixture providing a SemanticMatcher with mock embedder."""
    return SemanticMatcher(
        embedder=mock_embedder,
        # Thresholds tuned for mock embedder clusters
        causal_threshold=0.70,
        assertion_threshold=0.70,
        query_threshold=0.70,
        conflict_threshold=0.70,
    )
