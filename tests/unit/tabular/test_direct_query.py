# tests/unit/tabular/test_direct_query.py
"""Tests for direct table query (fast path)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from fitz_sage.tabular.direct_query import (
    DirectTableQuery,
    compute_file_hash,
    is_table_file,
    parse_columns,
    read_headers,
)


class TestIsTableFile:
    """Tests for is_table_file function."""

    def test_csv_is_table_file(self):
        assert is_table_file(Path("data.csv")) is True

    def test_tsv_is_table_file(self):
        assert is_table_file(Path("data.tsv")) is True

    def test_xlsx_is_table_file(self):
        assert is_table_file(Path("data.xlsx")) is True

    def test_xls_is_table_file(self):
        assert is_table_file(Path("data.xls")) is True

    def test_txt_is_not_table_file(self):
        assert is_table_file(Path("data.txt")) is False

    def test_pdf_is_not_table_file(self):
        assert is_table_file(Path("data.pdf")) is False

    def test_case_insensitive(self):
        assert is_table_file(Path("data.CSV")) is True
        assert is_table_file(Path("data.XLSX")) is True


class TestReadHeaders:
    """Tests for read_headers function."""

    def test_reads_csv_headers(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA")

        headers = read_headers(csv_file)

        assert headers == ["name", "age", "city"]

    def test_reads_tsv_headers(self, tmp_path):
        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text("name\tage\tcity\nAlice\t30\tNYC")

        headers = read_headers(tsv_file)

        assert headers == ["name", "age", "city"]

    def test_handles_empty_file(self, tmp_path):
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("")

        headers = read_headers(csv_file)

        assert headers == []


class TestParseColumns:
    """Tests for parse_columns function."""

    def test_parses_selected_columns_only(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age,city,country\nAlice,30,NYC,USA\nBob,25,LA,USA")

        headers, rows = parse_columns(csv_file, ["name", "city"])

        assert headers == ["name", "city"]
        assert rows == [["Alice", "NYC"], ["Bob", "LA"]]

    def test_preserves_column_order(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c,d\n1,2,3,4")

        headers, rows = parse_columns(csv_file, ["c", "a"])

        # Order matches the request, not the file
        assert headers == ["c", "a"]
        assert rows == [["3", "1"]]

    def test_handles_missing_column(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,30")

        headers, rows = parse_columns(csv_file, ["name", "missing"])

        assert headers == ["name"]  # Only existing columns
        assert rows == [["Alice"]]

    def test_returns_empty_if_no_columns_found(self, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b,c\n1,2,3")

        headers, rows = parse_columns(csv_file, ["x", "y", "z"])

        assert headers == []
        assert rows == []


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_same_content_same_hash(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.write_text("name,age\nAlice,30")
        file2.write_text("name,age\nAlice,30")

        assert compute_file_hash(file1) == compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path):
        file1 = tmp_path / "file1.csv"
        file2 = tmp_path / "file2.csv"
        file1.write_text("name,age\nAlice,30")
        file2.write_text("name,age\nBob,25")

        assert compute_file_hash(file1) != compute_file_hash(file2)

    def test_hash_is_deterministic(self, tmp_path):
        file = tmp_path / "test.csv"
        file.write_text("data")

        hash1 = compute_file_hash(file)
        hash2 = compute_file_hash(file)

        assert hash1 == hash2


class TestDirectTableQuery:
    """Tests for DirectTableQuery class."""

    @pytest.fixture
    def mock_chat_factory(self):
        """Create a mock chat factory that returns a mock client."""
        chat = MagicMock()
        chat.chat = MagicMock(return_value='["name", "age"]')

        def factory(tier: str = "fast"):
            return chat

        return factory, chat

    @pytest.fixture
    def sample_csv(self, tmp_path):
        """Create a sample CSV file."""
        csv_file = tmp_path / "people.csv"
        csv_file.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,SF")
        return csv_file

    def test_selects_columns_via_llm(self, mock_chat_factory, sample_csv):
        """Test that LLM is called to select columns."""
        factory, mock_chat = mock_chat_factory
        # Make chat return columns, then SQL, then answer
        mock_chat.chat = MagicMock(
            side_effect=[
                '["name", "age"]',  # Column selection
                'SELECT "name", "age" FROM "tbl_test" LIMIT 100',  # SQL
                "Alice is 30 years old",  # Answer
            ]
        )

        query = DirectTableQuery(chat_factory=factory)

        # Should call chat for column selection (now requires samples)
        samples = {"name": ["Alice", "Bob"], "age": ["30", "25"], "city": ["NYC", "LA"]}
        result = query._select_columns("Who is the oldest?", ["name", "age", "city"], samples)

        assert result == ["name", "age"]

    def test_parses_json_list_from_code_block(self, mock_chat_factory):
        """Test JSON parsing from markdown code block."""
        factory, _ = mock_chat_factory
        query = DirectTableQuery(chat_factory=factory)

        # fallback contains valid column names that match the parsed result
        result = query._parse_json_list('```json\n["a", "b"]\n```', fallback=["a", "b", "c"])

        assert result == ["a", "b"]

    def test_parses_plain_json_list(self, mock_chat_factory):
        """Test JSON parsing from plain response."""
        factory, _ = mock_chat_factory
        query = DirectTableQuery(chat_factory=factory)

        # fallback contains valid column names that match the parsed result
        result = query._parse_json_list('["col1", "col2"]', fallback=["col1", "col2", "col3"])

        assert result == ["col1", "col2"]

    def test_fallback_on_invalid_json(self, mock_chat_factory):
        """Test fallback when JSON is invalid."""
        factory, _ = mock_chat_factory
        query = DirectTableQuery(chat_factory=factory)

        result = query._parse_json_list("not valid json", fallback=["fallback"])

        assert result == ["fallback"]

    def test_extracts_sql_from_response(self, mock_chat_factory):
        """Test SQL extraction."""
        factory, _ = mock_chat_factory
        query = DirectTableQuery(chat_factory=factory)

        sql = query._extract_sql("```sql\nSELECT * FROM tbl\n```")

        assert sql == "SELECT * FROM tbl"

    def test_extracts_sql_without_code_block(self, mock_chat_factory):
        """Test SQL extraction from plain response."""
        factory, _ = mock_chat_factory
        query = DirectTableQuery(chat_factory=factory)

        sql = query._extract_sql('SELECT "name" FROM "people" LIMIT 10')

        assert sql == 'SELECT "name" FROM "people" LIMIT 10'

    def test_is_table_file_check(self, mock_chat_factory, tmp_path):
        """Test that non-table files are rejected."""
        factory, _ = mock_chat_factory
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("not a table")

        query = DirectTableQuery(chat_factory=factory)

        with pytest.raises(ValueError, match="Not a supported table file"):
            query.query(txt_file, "question")

    def test_file_not_found(self, mock_chat_factory, tmp_path):
        """Test that missing files raise error."""
        factory, _ = mock_chat_factory
        missing = tmp_path / "missing.csv"

        query = DirectTableQuery(chat_factory=factory)

        with pytest.raises(FileNotFoundError):
            query.query(missing, "question")
