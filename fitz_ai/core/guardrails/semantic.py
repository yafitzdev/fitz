# fitz_ai/core/guardrails/semantic.py
"""
Semantic Matcher - Language-agnostic concept detection using embeddings.

This module provides semantic matching capabilities for guardrails,
replacing brittle regex patterns with embedding-based similarity.

The key insight: multilingual embedding models map semantically similar
concepts to nearby vectors regardless of language. "because" (English),
"parce que" (French), "因为" (Chinese) all cluster together.

Usage:
    from fitz_ai.core.guardrails.semantic import SemanticMatcher

    matcher = SemanticMatcher(embedder)
    if matcher.has_causal_language(chunk.content):
        ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from fitz_ai.core.chunk import ChunkLike

# Type alias for embedder function
EmbedderFunc = Callable[[str], list[float]]


# =============================================================================
# Concept Definitions
# =============================================================================
# These are "anchor phrases" that define concepts. The embedder will map
# semantically similar phrases in ANY language to nearby vectors.

CAUSAL_CONCEPTS: tuple[str, ...] = (
    "because of this",
    "this was caused by",
    "the reason is",
    "this led to",
    "as a result of",
    "due to",
    "this happened because",
    "the cause was",
    "consequently",
    "therefore this occurred",
)

ASSERTION_CONCEPTS: tuple[str, ...] = (
    "this is definitely",
    "it was confirmed that",
    "the answer is",
    "this states that",
    "according to this",
    "it is known that",
    "the fact is",
    "this proves that",
)

CAUSAL_QUERY_CONCEPTS: tuple[str, ...] = (
    "why did this happen",
    "what caused this",
    "what is the reason",
    "explain why",
    "how did this occur",
    "what led to this",
)

FACT_QUERY_CONCEPTS: tuple[str, ...] = (
    "what is the answer",
    "which one is correct",
    "who is responsible",
    "where is this located",
    "when did this happen",
    "what is the value",
)

RESOLUTION_QUERY_CONCEPTS: tuple[str, ...] = (
    "which one is authoritative",
    "which source should I trust",
    "how to resolve this conflict",
    "which is the correct version",
    "reconcile these differences",
    "why do these disagree",
)

# Opposing concept pairs for conflict detection
# Each tuple contains (concept_a, concept_b) that are mutually exclusive
OPPOSING_CONCEPTS: tuple[tuple[str, str], ...] = (
    # State oppositions
    ("this was successful and completed", "this failed and did not complete"),
    ("this was approved and accepted", "this was rejected and denied"),
    ("this is active and enabled", "this is inactive and disabled"),
    ("this is confirmed and verified", "this is unconfirmed and unverified"),
    # Trend oppositions
    ("this improved and increased", "this declined and decreased"),
    ("this grew and expanded", "this shrank and contracted"),
    ("this remained stable and unchanged", "this changed significantly"),
    # Sentiment oppositions
    ("this is positive and good", "this is negative and bad"),
    ("the outcome was excellent", "the outcome was poor and terrible"),
    # Classification oppositions
    ("this is a security incident", "this is an operational incident"),
    ("this is internal", "this is external"),
    ("this is primary and main", "this is secondary and auxiliary"),
)


# =============================================================================
# Vector Math
# =============================================================================


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns value in [-1, 1] where 1 means identical direction.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute element-wise mean of vectors."""
    if not vectors:
        raise ValueError("Cannot compute mean of empty vector list")

    dim = len(vectors[0])
    result = [0.0] * dim

    for vec in vectors:
        for i, val in enumerate(vec):
            result[i] += val

    n = len(vectors)
    return [v / n for v in result]


# =============================================================================
# Semantic Matcher
# =============================================================================


@dataclass
class SemanticMatcher:
    """
    Language-agnostic semantic concept detection using embeddings.

    This class replaces regex-based pattern matching with embedding
    similarity, enabling robust detection across any language.

    Concept vectors are lazily computed and cached for efficiency.

    Attributes:
        embedder: Function that converts text to embedding vector
        causal_threshold: Similarity threshold for causal language detection
        assertion_threshold: Similarity threshold for assertion detection
        query_threshold: Similarity threshold for query type classification
        conflict_threshold: Similarity threshold for conflict detection
    """

    embedder: EmbedderFunc
    causal_threshold: float = 0.65
    assertion_threshold: float = 0.60
    query_threshold: float = 0.65
    conflict_threshold: float = 0.70

    # Internal caches (not part of dataclass comparison)
    _concept_cache: dict[str, list[list[float]]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _centroid_cache: dict[str, list[float]] = field(default_factory=dict, repr=False, compare=False)

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _get_concept_vectors(self, key: str, concepts: tuple[str, ...]) -> list[list[float]]:
        """Get cached concept vectors or compute and cache them."""
        if key not in self._concept_cache:
            self._concept_cache[key] = [self.embedder(c) for c in concepts]
        return self._concept_cache[key]

    def _get_centroid(self, key: str, concepts: tuple[str, ...]) -> list[float]:
        """Get cached centroid vector or compute and cache it."""
        if key not in self._centroid_cache:
            vectors = self._get_concept_vectors(key, concepts)
            self._centroid_cache[key] = mean_vector(vectors)
        return self._centroid_cache[key]

    def _embed_text(self, text: str) -> list[float]:
        """Embed text using the configured embedder."""
        return self.embedder(text)

    # -------------------------------------------------------------------------
    # Similarity Computation
    # -------------------------------------------------------------------------

    def max_similarity_to_concepts(
        self,
        text: str,
        concept_key: str,
        concepts: tuple[str, ...],
    ) -> float:
        """
        Find maximum cosine similarity between text and any concept.

        This is more sensitive than centroid comparison - useful when
        any single concept match is meaningful.
        """
        text_vec = self._embed_text(text)
        concept_vecs = self._get_concept_vectors(concept_key, concepts)

        return max(cosine_similarity(text_vec, cv) for cv in concept_vecs)

    def similarity_to_centroid(
        self,
        text: str,
        concept_key: str,
        concepts: tuple[str, ...],
    ) -> float:
        """
        Compute similarity between text and concept centroid.

        Centroid comparison is faster (single comparison) and less
        sensitive to individual concept variations.
        """
        text_vec = self._embed_text(text)
        centroid = self._get_centroid(concept_key, concepts)

        return cosine_similarity(text_vec, centroid)

    # -------------------------------------------------------------------------
    # Query Type Detection
    # -------------------------------------------------------------------------

    def is_causal_query(self, query: str) -> bool:
        """
        Detect if query asks for causal explanation.

        Works across languages - "why did this happen", "pourquoi",
        "为什么" all detected as causal queries.
        """
        similarity = self.max_similarity_to_concepts(query, "causal_query", CAUSAL_QUERY_CONCEPTS)
        return similarity >= self.query_threshold

    def is_fact_query(self, query: str) -> bool:
        """
        Detect if query asks for factual information.

        Works across languages for who/what/where/when type questions.
        """
        similarity = self.max_similarity_to_concepts(query, "fact_query", FACT_QUERY_CONCEPTS)
        return similarity >= self.query_threshold

    def is_resolution_query(self, query: str) -> bool:
        """
        Detect if query explicitly asks for conflict resolution.

        Queries like "Which source should I trust?" should allow
        decisive answers even when conflicts exist.
        """
        similarity = self.max_similarity_to_concepts(
            query, "resolution_query", RESOLUTION_QUERY_CONCEPTS
        )
        return similarity >= self.query_threshold

    # -------------------------------------------------------------------------
    # Evidence Detection
    # -------------------------------------------------------------------------

    def has_causal_language(self, text: str) -> bool:
        """
        Check if text contains causal language.

        Detects "because", "due to", etc. in any language.
        """
        similarity = self.max_similarity_to_concepts(text, "causal", CAUSAL_CONCEPTS)
        return similarity >= self.causal_threshold

    def has_assertion(self, text: str) -> bool:
        """
        Check if text contains definitive assertions.

        Detects "is", "was confirmed", etc. in any language.
        """
        similarity = self.max_similarity_to_concepts(text, "assertion", ASSERTION_CONCEPTS)
        return similarity >= self.assertion_threshold

    def count_causal_chunks(self, chunks: Sequence["ChunkLike"]) -> int:
        """Count chunks containing causal language."""
        return sum(1 for chunk in chunks if self.has_causal_language(chunk.content))

    def count_assertion_chunks(self, chunks: Sequence["ChunkLike"]) -> int:
        """Count chunks containing assertions."""
        return sum(1 for chunk in chunks if self.has_assertion(chunk.content))

    # -------------------------------------------------------------------------
    # Conflict Detection
    # -------------------------------------------------------------------------

    def _get_opposing_vectors(self) -> list[tuple[list[float], list[float]]]:
        """Get cached opposing concept vector pairs."""
        if "opposing_pairs" not in self._concept_cache:
            pairs = []
            for concept_a, concept_b in OPPOSING_CONCEPTS:
                vec_a = self.embedder(concept_a)
                vec_b = self.embedder(concept_b)
                pairs.append((vec_a, vec_b))
            self._concept_cache["opposing_pairs"] = pairs
        return self._concept_cache["opposing_pairs"]

    def detect_conflict(
        self,
        text_a: str,
        text_b: str,
    ) -> tuple[bool, str | None]:
        """
        Detect if two texts make conflicting claims.

        Uses opposing concept pairs to check if text_a is similar to
        one side of an opposition while text_b is similar to the other.

        Returns:
            (is_conflict, conflict_type) - conflict_type describes the
            nature of the conflict if found, None otherwise.
        """
        vec_a = self._embed_text(text_a)
        vec_b = self._embed_text(text_b)
        opposing_pairs = self._get_opposing_vectors()

        for i, (concept_pos, concept_neg) in enumerate(opposing_pairs):
            # Check if text_a aligns with positive and text_b with negative
            sim_a_pos = cosine_similarity(vec_a, concept_pos)
            sim_a_neg = cosine_similarity(vec_a, concept_neg)
            sim_b_pos = cosine_similarity(vec_b, concept_pos)
            sim_b_neg = cosine_similarity(vec_b, concept_neg)

            # Conflict if one text strongly aligns with one side and
            # the other text strongly aligns with the opposite side
            a_is_positive = sim_a_pos >= self.conflict_threshold and sim_a_pos > sim_a_neg
            a_is_negative = sim_a_neg >= self.conflict_threshold and sim_a_neg > sim_a_pos
            b_is_positive = sim_b_pos >= self.conflict_threshold and sim_b_pos > sim_b_neg
            b_is_negative = sim_b_neg >= self.conflict_threshold and sim_b_neg > sim_b_pos

            if (a_is_positive and b_is_negative) or (a_is_negative and b_is_positive):
                conflict_type = f"{OPPOSING_CONCEPTS[i][0]} vs {OPPOSING_CONCEPTS[i][1]}"
                return True, conflict_type

        return False, None

    def find_conflicts(
        self,
        chunks: Sequence["ChunkLike"],
    ) -> list[tuple[str, str, str]]:
        """
        Find all conflicting claim pairs across chunks.

        Args:
            chunks: Sequence of chunks to analyze

        Returns:
            List of (chunk_id_a, chunk_id_b, conflict_type) tuples
        """
        conflicts: list[tuple[str, str, str]] = []

        # Compare all pairs
        chunk_list = list(chunks)
        for i in range(len(chunk_list)):
            for j in range(i + 1, len(chunk_list)):
                chunk_a = chunk_list[i]
                chunk_b = chunk_list[j]

                is_conflict, conflict_type = self.detect_conflict(chunk_a.content, chunk_b.content)

                if is_conflict and conflict_type:
                    conflicts.append((chunk_a.id, chunk_b.id, conflict_type))

        return conflicts


__all__ = [
    "SemanticMatcher",
    "EmbedderFunc",
    "cosine_similarity",
    "mean_vector",
    "CAUSAL_CONCEPTS",
    "ASSERTION_CONCEPTS",
    "CAUSAL_QUERY_CONCEPTS",
    "FACT_QUERY_CONCEPTS",
    "RESOLUTION_QUERY_CONCEPTS",
    "OPPOSING_CONCEPTS",
]
