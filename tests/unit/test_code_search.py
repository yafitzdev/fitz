# tests/unit/test_code_search.py
"""Tests for CodeSearchStrategy: keyword + BM25 search."""

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
        top_addresses=10,
    )


@pytest.fixture
def mock_symbol_store():
    return MagicMock()


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

    def test_deduplication(self, mock_symbol_store, config):
        sym = _make_symbol("s1", "func", "mod.func")
        mock_symbol_store.search_by_name.return_value = [sym]
        mock_symbol_store.search_bm25.return_value = [{**sym, "bm25_score": 0.8}]
        mock_symbol_store.search_by_keywords.return_value = []

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("func", limit=5)

        # Same symbol from both legs should be merged, not duplicated
        assert len(results) == 1

    def test_respects_limit(self, mock_symbol_store, config):
        syms = [_make_symbol(f"s{i}", f"func{i}", f"mod.func{i}") for i in range(20)]
        mock_symbol_store.search_by_name.return_value = syms
        mock_symbol_store.search_bm25.return_value = []
        mock_symbol_store.search_by_keywords.return_value = []

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("func", limit=3)

        assert len(results) == 3

    def test_address_metadata(self, mock_symbol_store, config):
        mock_symbol_store.search_by_name.return_value = [
            _make_symbol("s1", "my_func", "pkg.my_func"),
        ]
        mock_symbol_store.search_bm25.return_value = []
        mock_symbol_store.search_by_keywords.return_value = []

        strategy = CodeSearchStrategy(mock_symbol_store, config)
        results = strategy.retrieve("my_func", limit=5)

        addr = results[0]
        assert addr.metadata["name"] == "my_func"
        assert addr.metadata["qualified_name"] == "pkg.my_func"
        assert addr.metadata["start_line"] == 1
        assert addr.metadata["end_line"] == 10
