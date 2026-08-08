# tests/unit/test_code_expander.py
"""Tests for CodeExpander: import expansion, class context, deduplication."""

from unittest.mock import MagicMock

import pytest

from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.retrieval.expander import CodeExpander
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult


@pytest.fixture
def config():
    return FitzKragConfig(
        collection="test",
        max_expansion_depth=1,
        include_class_context=True,
    )


@pytest.fixture
def mock_stores():
    raw_store = MagicMock()
    symbol_store = MagicMock()
    import_store = MagicMock()
    return raw_store, symbol_store, import_store


def _make_symbol_result(name="func", kind="function", class_name=None):
    qualified = f"mod.{class_name}.{name}" if class_name else f"mod.{name}"
    addr = Address(
        kind=AddressKind.SYMBOL,
        source_id="f1",
        location=qualified,
        summary=f"Summary for {name}",
        metadata={
            "kind": kind,
            "qualified_name": qualified,
            "start_line": 5,
            "end_line": 10,
        },
    )
    return ReadResult(
        address=addr,
        content=f"def {name}():\n    pass",
        file_path="src/mod.py",
        line_range=(5, 10),
    )


class TestCodeExpander:
    def test_no_expansion_when_depth_zero(self, mock_stores):
        config = FitzKragConfig(collection="test", max_expansion_depth=0)
        raw_store, symbol_store, import_store = mock_stores
        expander = CodeExpander(raw_store, symbol_store, import_store, config)

        result = _make_symbol_result()
        expanded = expander.expand([result])
        assert len(expanded) == 1

    def test_adds_file_imports(self, mock_stores, config):
        raw_store, symbol_store, import_store = mock_stores
        raw_store.get.return_value = {
            "id": "f1",
            "path": "src/mod.py",
            "content": "import os\nfrom pathlib import Path\n\ndef func():\n    pass\n",
        }
        expander = CodeExpander(raw_store, symbol_store, import_store, config)

        result = _make_symbol_result()
        expanded = expander.expand([result])

        # Should have original + imports block
        assert len(expanded) == 2
        import_block = [r for r in expanded if r.metadata.get("context_type") == "imports"]
        assert len(import_block) == 1
        assert "import os" in import_block[0].content

    def test_no_duplicate_imports(self, mock_stores, config):
        raw_store, symbol_store, import_store = mock_stores
        raw_store.get.return_value = {
            "id": "f1",
            "path": "src/mod.py",
            "content": "import os\n\ndef func1():\n    pass\n\ndef func2():\n    pass\n",
        }
        expander = CodeExpander(raw_store, symbol_store, import_store, config)

        results = [
            _make_symbol_result("func1"),
            _make_symbol_result("func2"),
        ]
        expanded = expander.expand(results)

        import_blocks = [r for r in expanded if r.metadata.get("context_type") == "imports"]
        # Same file, same imports — should only appear once
        assert len(import_blocks) == 1

    def test_adds_class_context_for_methods(self, mock_stores, config):
        raw_store, symbol_store, import_store = mock_stores
        raw_store.get.return_value = {
            "id": "f1",
            "path": "src/mod.py",
            "content": "import os\n\nclass MyClass:\n    '''Doc.'''\n    def method(self):\n        pass\n",
        }
        symbol_store.search_by_name.return_value = [
            {
                "id": "cls1",
                "name": "MyClass",
                "qualified_name": "mod.MyClass",
                "kind": "class",
                "raw_file_id": "f1",
                "start_line": 3,
                "end_line": 6,
                "signature": "class MyClass",
                "summary": "A class",
                "metadata": {},
            }
        ]
        expander = CodeExpander(raw_store, symbol_store, import_store, config)

        result = _make_symbol_result("method", kind="method", class_name="MyClass")
        expanded = expander.expand([result])

        class_headers = [r for r in expanded if r.metadata.get("context_type") == "class_header"]
        assert len(class_headers) == 1
        assert "class MyClass" in class_headers[0].content

    def test_skips_non_symbol_addresses(self, mock_stores, config):
        raw_store, symbol_store, import_store = mock_stores
        expander = CodeExpander(raw_store, symbol_store, import_store, config)

        addr = Address(kind=AddressKind.SECTION, source_id="s1", location="doc", summary="")
        result = ReadResult(address=addr, content="section text", file_path="doc.pdf")
        expanded = expander.expand([result])
        assert len(expanded) == 1

    def test_deduplication(self, mock_stores, config):
        raw_store, symbol_store, import_store = mock_stores
        raw_store.get.return_value = {
            "id": "f1",
            "path": "src/mod.py",
            "content": "import os\n\ndef func():\n    pass\n",
        }
        expander = CodeExpander(raw_store, symbol_store, import_store, config)

        # Same result twice
        result = _make_symbol_result()
        expanded = expander.expand([result, result])

        # Dedup should remove the duplicate
        unique_locations = set()
        for r in expanded:
            key = (r.file_path, r.address.location, r.line_range[0] if r.line_range else None)
            unique_locations.add(key)
        assert len(unique_locations) == len(expanded)
