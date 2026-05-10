# tests/unit/test_code_search.py
"""Tests for CodeSearchStrategy: keyword + semantic search, hybrid merge."""

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.retrieval.strategies.code_search import CodeSearchStrategy
from fitz_sage.engines.fitz_krag.types import AddressKind


@pytest.fixture
def config():
    return FitzKragConfig(
        collection="test",
        keyword_weight=0.4,
        semantic_weight=0.6,
        top_addresses=10,
    )


@pytest.fixture
def mock_symbol_store():
    return MagicMock()


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    return embedder


def _make_symbol(sid, name, qualified, kind="function"):
    return {
        "id": sid,
        "name": name,
        "qualified_name": qualified,
        "kind": kind,
        "raw_file_id": f"file_{sid}",
        "start_line": 1,
        "end_line": 10,
        "signature": f"def {name}()",
        "summary": f"Summary for {name}",
        "metadata": {},
    }


class TestCodeSearchStrategy:
    def test_retrieve_combines_keyword_and_bm25(self, mock_symbol_store, config):
        mock_symbol_store.search_by_name.return_value = [
            _make_symbol("s1", "process", "mod.process"),
        ]
        mock_symbol_store.search_bm25.return_value = [
            {**_make_symbol("s2", "transform", "mod.transform"), "bm25_score": 0.9},
        ]
        mock_symbol_store.search_by_keywords.return_value = []

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("process data", limit=5)

        assert len(results) == 2
        assert all(r.kind == AddressKind.SYMBOL for r in results)

    def test_keyword_only_on_semantic_failure(self, mock_symbol_store, mock_embedder, config):
        mock_symbol_store.search_by_name.return_value = [
            _make_symbol("s1", "func", "mod.func"),
        ]
        mock_embedder.embed.side_effect = Exception("API error")

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("func", limit=5)

        assert len(results) == 1
        assert results[0].metadata["name"] == "func"

    def test_deduplication(self, mock_symbol_store, mock_embedder, config):
        sym = _make_symbol("s1", "func", "mod.func")
        mock_symbol_store.search_by_name.return_value = [sym]
        mock_symbol_store.search_by_vector.return_value = [{**sym, "score": 0.8}]

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("func", limit=5)

        # Same symbol from both searches should be merged, not duplicated
        assert len(results) == 1

    def test_respects_limit(self, mock_symbol_store, mock_embedder, config):
        syms = [_make_symbol(f"s{i}", f"func{i}", f"mod.func{i}") for i in range(20)]
        mock_symbol_store.search_by_name.return_value = syms
        mock_symbol_store.search_by_vector.return_value = []

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("func", limit=3)

        assert len(results) == 3

    def test_address_metadata(self, mock_symbol_store, mock_embedder, config):
        mock_symbol_store.search_by_name.return_value = [
            _make_symbol("s1", "my_func", "pkg.my_func"),
        ]
        mock_symbol_store.search_by_vector.return_value = []

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("my_func", limit=5)

        addr = results[0]
        assert addr.metadata["name"] == "my_func"
        assert addr.metadata["qualified_name"] == "pkg.my_func"
        assert addr.metadata["start_line"] == 1
        assert addr.metadata["end_line"] == 10
