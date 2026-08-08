# tests/unit/test_krag_types.py
"""Tests for KRAG core types: Address, AddressKind, ReadResult."""

import pytest

from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult


class TestAddressKind:
    def test_enum_values(self):
        assert AddressKind.SYMBOL == "symbol"
        assert AddressKind.FILE == "file"
        assert AddressKind.SECTION == "section"
        assert AddressKind.TABLE == "table"

    def test_string_enum(self):
        assert isinstance(AddressKind.SYMBOL, str)
        assert AddressKind.SYMBOL.value == "symbol"


class TestAddress:
    def test_creation(self):
        addr = Address(
            kind=AddressKind.SYMBOL,
            source_id="file_123",
            location="module.MyClass.my_method",
            summary="Does something useful",
        )
        assert addr.kind == AddressKind.SYMBOL
        assert addr.source_id == "file_123"
        assert addr.location == "module.MyClass.my_method"
        assert addr.summary == "Does something useful"
        assert addr.score == 0.0
        assert addr.metadata == {}

    def test_with_metadata(self):
        addr = Address(
            kind=AddressKind.SYMBOL,
            source_id="f1",
            location="mod.func",
            summary="A function",
            score=0.95,
            metadata={"start_line": 10, "end_line": 25},
        )
        assert addr.score == 0.95
        assert addr.metadata["start_line"] == 10

    def test_frozen(self):
        addr = Address(
            kind=AddressKind.SYMBOL,
            source_id="f1",
            location="x",
            summary="y",
        )
        with pytest.raises(AttributeError):
            addr.kind = AddressKind.FILE  # type: ignore[misc]

    def test_equality(self):
        addr1 = Address(kind=AddressKind.SYMBOL, source_id="f1", location="x", summary="y")
        addr2 = Address(kind=AddressKind.SYMBOL, source_id="f1", location="x", summary="y")
        assert addr1 == addr2


class TestReadResult:
    def test_creation(self):
        addr = Address(
            kind=AddressKind.SYMBOL,
            source_id="f1",
            location="mod.func",
            summary="test",
        )
        result = ReadResult(
            address=addr,
            content="def func(): pass",
            file_path="src/mod.py",
            line_range=(5, 10),
        )
        assert result.content == "def func(): pass"
        assert result.file_path == "src/mod.py"
        assert result.line_range == (5, 10)

    def test_no_line_range(self):
        addr = Address(kind=AddressKind.FILE, source_id="f1", location="f.py", summary="")
        result = ReadResult(address=addr, content="content", file_path="f.py")
        assert result.line_range is None
