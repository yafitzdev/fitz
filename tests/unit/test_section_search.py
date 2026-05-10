# tests/unit/test_section_search.py
"""Tests for SectionSearchStrategy — BM25 + semantic hybrid retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.retrieval.strategies.section_search import (
    SectionSearchStrategy,
)
from fitz_sage.engines.fitz_krag.types import AddressKind


@pytest.fixture
def mock_section_store():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.section_bm25_weight = 0.6
    config.section_semantic_weight = 0.4
    return config


@pytest.fixture
def strategy(mock_section_store, mock_embedder, mock_config):
    return SectionSearchStrategy(mock_section_store, mock_config)


def _make_section_result(
    id_="sec1",
    raw_file_id="file1",
    title="Introduction",
    level=1,
    page_start=1,
    page_end=3,
    content="Some content.",
    summary="Section summary.",
    parent_section_id=None,
    position=0,
    metadata=None,
    bm25_score=None,
    score=None,
):
    d = {
        "id": id_,
        "raw_file_id": raw_file_id,
        "title": title,
        "level": level,
        "page_start": page_start,
        "page_end": page_end,
        "content": content,
        "summary": summary,
        "parent_section_id": parent_section_id,
        "position": position,
        "metadata": metadata or {},
    }
    if bm25_score is not None:
        d["bm25_score"] = bm25_score
    if score is not None:
        d["score"] = score
    return d


class TestRetrieve:
    def test_returns_addresses_with_section_kind(self, strategy, mock_section_store, mock_embedder):
        mock_section_store.search_bm25.return_value = [
            _make_section_result(bm25_score=0.9),
        ]
        mock_section_store.search_by_vector.return_value = []

        results = strategy.retrieve("introduction", limit=5)
        assert len(results) == 1
        assert results[0].kind == AddressKind.SECTION

    def test_address_contains_section_metadata(self, strategy, mock_section_store, mock_embedder):
        mock_section_store.search_bm25.return_value = [
            _make_section_result(
                id_="sec1",
                raw_file_id="file1",
                page_start=5,
                page_end=8,
                level=2,
                parent_section_id="parent1",
                bm25_score=0.8,
            ),
        ]
        mock_section_store.search_by_vector.return_value = []
        # Parent lookup returns the parent section with its title
        mock_section_store.get.return_value = _make_section_result(
            id_="parent1", title="Parent Section"
        )

        results = strategy.retrieve("query", limit=5)
        addr = results[0]
        assert addr.source_id == "file1"
        assert addr.location == "Parent Section > Introduction"
        assert addr.metadata["section_id"] == "sec1"
        assert addr.metadata["level"] == 2
        assert addr.metadata["page_start"] == 5
        assert addr.metadata["page_end"] == 8
        assert addr.metadata["parent_section_id"] == "parent1"

    def test_semantic_fallback_on_embed_failure(self, strategy, mock_section_store, mock_embedder):
        mock_section_store.search_bm25.return_value = [
            _make_section_result(bm25_score=0.7),
        ]
        mock_embedder.embed.side_effect = RuntimeError("Embed failed")

        results = strategy.retrieve("query", limit=5)
        # Should still return BM25 results
        assert len(results) == 1

    def test_respects_limit(self, strategy, mock_section_store, mock_embedder):
        mock_section_store.search_bm25.return_value = [
            _make_section_result(id_=f"sec{i}", bm25_score=0.9 - i * 0.1) for i in range(10)
        ]
        mock_section_store.search_by_vector.return_value = []

        results = strategy.retrieve("query", limit=3)
        assert len(results) == 3


class TestHybridMerge:
    def test_bm25_weighted_higher(self, strategy, mock_section_store, mock_embedder):
        # Same section appears in both BM25 and semantic
        mock_section_store.search_bm25.return_value = [
            _make_section_result(id_="sec1", bm25_score=1.0),
        ]
        mock_section_store.search_by_vector.return_value = [
            _make_section_result(id_="sec1", score=1.0),
        ]

        results = strategy.retrieve("query", limit=5)
        assert len(results) == 1
        # RRF with k=60: both legs at rank 0 → 1/(60+0) = 1/60
        # Combined: bm25_weight * (1/60) + semantic_weight * (1/60)
        #         = (0.6 + 0.4) * (1/60) = 1/60 ≈ 0.01667
        assert results[0].score == pytest.approx(1 / 60, abs=1e-4)

    def test_disjoint_results_merged(self, strategy, mock_section_store, mock_embedder):
        mock_section_store.search_bm25.return_value = [
            _make_section_result(id_="sec1", raw_file_id="f1", bm25_score=0.8),
        ]
        mock_section_store.search_by_vector.return_value = [
            _make_section_result(id_="sec2", raw_file_id="f2", score=0.9),
        ]

        results = strategy.retrieve("query", limit=5)
        assert len(results) == 2

    def test_bm25_only_when_no_semantic(self, strategy, mock_section_store, mock_embedder):
        mock_section_store.search_bm25.return_value = [
            _make_section_result(id_="sec1", bm25_score=0.5),
        ]
        mock_section_store.search_by_vector.return_value = []

        results = strategy.retrieve("query", limit=5)
        assert len(results) == 1
        # RRF with k=60: BM25 only at rank 0 → 1/(60+0) = 1/60
        # Score = bm25_weight * (1/60) = 0.6 * (1/60) = 0.6/60 = 0.01
        assert results[0].score == pytest.approx(0.6 / 60, abs=1e-4)


class TestToAddress:
    def test_summary_from_section_summary(self, strategy):
        section = _make_section_result(summary="Good summary.")
        section["combined_score"] = 0.8
        addr = strategy._to_address(section)
        assert addr.summary == "Good summary."

    def test_summary_falls_back_to_title(self, strategy):
        section = _make_section_result(summary=None, title="Fallback Title")
        section["combined_score"] = 0.5
        addr = strategy._to_address(section)
        assert addr.summary == "Fallback Title"
