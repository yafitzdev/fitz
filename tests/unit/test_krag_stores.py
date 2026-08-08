# tests/unit/test_krag_stores.py
"""Tests for KRAG stores (RawFileStore, SymbolStore, ImportGraphStore) with mocked DB."""

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore


@pytest.fixture
def mock_cm():
    """Mock connection manager with a mock connection context."""
    cm = MagicMock()
    conn = MagicMock()
    cm.connection.return_value.__enter__ = MagicMock(return_value=conn)
    cm.connection.return_value.__exit__ = MagicMock(return_value=False)
    return cm, conn


class TestRawFileStore:
    def test_upsert(self, mock_cm):
        cm, conn = mock_cm
        store = RawFileStore(cm, "test_col")
        store.upsert("f1", "path/to/file.py", "content", "hash123", ".py", 100)
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_get_found(self, mock_cm):
        cm, conn = mock_cm
        conn.execute.return_value.fetchone.return_value = (
            "f1",
            "path/to/file.py",
            "content",
            "hash123",
            ".py",
            100,
            {},
        )
        store = RawFileStore(cm, "test_col")
        result = store.get("f1")
        assert result is not None
        assert result["id"] == "f1"
        assert result["path"] == "path/to/file.py"

    def test_get_not_found(self, mock_cm):
        cm, conn = mock_cm
        conn.execute.return_value.fetchone.return_value = None
        store = RawFileStore(cm, "test_col")
        result = store.get("nonexistent")
        assert result is None

    def test_list_hashes(self, mock_cm):
        cm, conn = mock_cm
        conn.execute.return_value.fetchall.return_value = [
            ("file1.py", "hash1"),
            ("file2.py", "hash2"),
        ]
        store = RawFileStore(cm, "test_col")
        hashes = store.list_hashes()
        assert hashes == {"file1.py": "hash1", "file2.py": "hash2"}

    def test_delete(self, mock_cm):
        cm, conn = mock_cm
        store = RawFileStore(cm, "test_col")
        store.delete("f1")
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()


class TestSymbolStore:
    def test_upsert_batch(self, mock_cm):
        cm, conn = mock_cm
        store = SymbolStore(cm, "test_col")
        symbols = [
            {
                "id": "s1",
                "name": "func",
                "qualified_name": "mod.func",
                "kind": "function",
                "raw_file_id": "f1",
                "start_line": 1,
                "end_line": 5,
                "signature": "def func()",
                "imports": [],
                "references": [],
                "metadata": {},
            }
        ]
        store.upsert_batch(symbols)
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_upsert_batch_empty(self, mock_cm):
        cm, conn = mock_cm
        store = SymbolStore(cm, "test_col")
        store.upsert_batch([])
        conn.execute.assert_not_called()

    def test_search_by_name(self, mock_cm):
        cm, conn = mock_cm
        conn.execute.return_value.fetchall.return_value = [
            ("s1", "func", "mod.func", "function", "f1", 1, 5, "def func()", [], {})
        ]
        store = SymbolStore(cm, "test_col")
        results = store.search_by_name("func")
        assert len(results) == 1
        assert results[0]["name"] == "func"
        assert results[0]["entities"] == []

    def test_delete_by_file(self, mock_cm):
        cm, conn = mock_cm
        store = SymbolStore(cm, "test_col")
        store.delete_by_file("f1")
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_get_by_file(self, mock_cm):
        """Returns symbols with references, ordered by start_line."""
        cm, conn = mock_cm
        conn.execute.return_value.fetchall.return_value = [
            (
                "s1",
                "helper",
                "mod.helper",
                "function",
                "f1",
                1,
                3,
                "def helper()",
                [{"name": "os", "type": "module"}],
                {},
                ["os", "sys"],
            ),
            (
                "s2",
                "main_fn",
                "mod.main_fn",
                "function",
                "f1",
                5,
                10,
                "def main_fn()",
                [],
                {},
                ["helper"],
            ),
        ]
        store = SymbolStore(cm, "test_col")
        results = store.get_by_file("f1")
        assert len(results) == 2
        assert results[0]["name"] == "helper"
        assert results[0]["entities"] == [{"name": "os", "type": "module"}]
        assert results[0]["references"] == ["os", "sys"]
        assert results[1]["name"] == "main_fn"
        assert results[1]["references"] == ["helper"]

    def test_get_by_file_empty(self, mock_cm):
        """Returns empty list for nonexistent file."""
        cm, conn = mock_cm
        conn.execute.return_value.fetchall.return_value = []
        store = SymbolStore(cm, "test_col")
        results = store.get_by_file("nonexistent")
        assert results == []


class TestImportGraphStore:
    def test_upsert_batch(self, mock_cm):
        cm, conn = mock_cm
        store = ImportGraphStore(cm, "test_col")
        edges = [
            {
                "source_file_id": "f1",
                "target_module": "os",
                "target_file_id": None,
                "import_names": ["path"],
            }
        ]
        store.upsert_batch(edges)
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_get_imports(self, mock_cm):
        cm, conn = mock_cm
        conn.execute.return_value.fetchall.return_value = [
            ("f1", "os", None, ["path"]),
            ("f1", "sys", None, []),
        ]
        store = ImportGraphStore(cm, "test_col")
        imports = store.get_imports("f1")
        assert len(imports) == 2
        assert imports[0]["target_module"] == "os"

    def test_get_importers(self, mock_cm):
        cm, conn = mock_cm
        conn.execute.return_value.fetchall.return_value = [
            ("f2", "mymod", "f1", ["func"]),
        ]
        store = ImportGraphStore(cm, "test_col")
        importers = store.get_importers("f1")
        assert len(importers) == 1
        assert importers[0]["source_file_id"] == "f2"

    def test_delete_by_file(self, mock_cm):
        cm, conn = mock_cm
        store = ImportGraphStore(cm, "test_col")
        store.delete_by_file("f1")
        conn.execute.assert_called_once()

    def test_resolve_targets(self, mock_cm):
        """Resolves 2/3 edges (one unresolvable stdlib import)."""
        cm, conn = mock_cm
        # SELECT returns 3 unresolved edges
        conn.execute.return_value.fetchall.return_value = [
            ("f1", "mypackage.utils"),
            ("f1", "mypackage.models"),
            ("f1", "os"),  # stdlib — won't resolve
        ]
        store = ImportGraphStore(cm, "test_col")
        path_to_id = {
            "mypackage/utils.py": "f2",
            "mypackage/models.py": "f3",
        }
        resolved = store.resolve_targets(path_to_id)
        assert resolved == 2
        # SELECT (1) + UPDATE (2) = 3 execute calls
        assert conn.execute.call_count == 3

    def test_resolve_targets_no_unresolved(self, mock_cm):
        """Returns 0 when no unresolved edges exist."""
        cm, conn = mock_cm
        conn.execute.return_value.fetchall.return_value = []
        store = ImportGraphStore(cm, "test_col")
        resolved = store.resolve_targets({"foo.py": "f1"})
        assert resolved == 0
