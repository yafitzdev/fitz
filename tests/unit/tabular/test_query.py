# tests/tabular/test_query.py
"""Tests for TableQueryStep - table queries at retrieval time."""

from unittest.mock import MagicMock

import pytest

from fitz_sage.core.chunk import Chunk
from fitz_sage.tabular.models import ParsedTable, create_schema_chunk
from fitz_sage.tabular.query import TableQueryStep
from fitz_sage.tabular.sql_gen import extract_sql


def create_mock_chat_factory():
    """Create a mock chat factory that returns the same mock client."""
    mock_chat = MagicMock()

    def factory(tier: str = "fast"):
        return mock_chat

    return factory, mock_chat


class MockTableStore:
    """Mock SqliteTableStore for testing."""

    def __init__(self):
        self._tables: dict[str, dict] = {}

    def add_table(
        self,
        table_id: str,
        pg_name: str,
        sanitized_cols: list[str],
        original_cols: list[str],
        rows: list[list[str]],
    ):
        """Add a table to the mock store."""
        self._tables[table_id] = {
            "pg_name": pg_name,
            "sanitized_cols": sanitized_cols,
            "original_cols": original_cols,
            "rows": rows,
            "row_count": len(rows),
        }

    def get_table_name(self, table_id: str) -> str | None:
        if table_id in self._tables:
            return self._tables[table_id]["pg_name"]
        return None

    def get_columns(self, table_id: str) -> tuple[list[str], list[str]] | None:
        if table_id in self._tables:
            t = self._tables[table_id]
            return t["sanitized_cols"], t["original_cols"]
        return None

    def get_row_count(self, table_id: str) -> int | None:
        if table_id in self._tables:
            return self._tables[table_id]["row_count"]
        return None

    def execute_query(
        self, table_id: str, sql: str, params: tuple = ()
    ) -> tuple[list[str], list[list]] | None:
        """Mock SQL execution - just return all rows for simplicity."""
        # Find the table from SQL (crude but works for tests)
        for tid, t in self._tables.items():
            if t["pg_name"] in sql:
                return t["sanitized_cols"], t["rows"]
        return None

    def execute_multi_table_query(
        self, sql: str, params: tuple = ()
    ) -> tuple[list[str], list[list]] | None:
        """Mock multi-table query execution."""
        # For testing, just return first table's data
        if self._tables:
            t = list(self._tables.values())[0]
            return t["sanitized_cols"], t["rows"]
        return None


class TestTableQueryStep:
    """Tests for TableQueryStep."""

    @pytest.fixture
    def mock_chat_factory(self):
        """Create a mock chat factory."""
        return create_mock_chat_factory()

    @pytest.fixture
    def mock_store(self):
        """Create a mock table store."""
        store = MockTableStore()
        store.add_table(
            table_id="test123",
            pg_name="tbl_test123",
            sanitized_cols=["country", "population", "gdp", "continent"],
            original_cols=["Country", "Population", "GDP", "Continent"],
            rows=[
                ["USA", "330000000", "21000", "North America"],
                ["China", "1400000000", "14000", "Asia"],
                ["Germany", "83000000", "3800", "Europe"],
                ["Brazil", "210000000", "1800", "South America"],
            ],
        )
        return store

    @pytest.fixture
    def step(self, mock_chat_factory, mock_store):
        """Create step with mock chat factory and store."""
        factory, _ = mock_chat_factory
        return TableQueryStep(chat_factory=factory, table_store=mock_store)

    @pytest.fixture
    def sample_table_chunk(self):
        """Create a sample table schema chunk."""
        table = ParsedTable(
            table_id="test123",
            source_doc="data.csv",
            headers=["Country", "Population", "GDP", "Continent"],
            rows=[
                ["USA", "330000000", "21000", "North America"],
                ["China", "1400000000", "14000", "Asia"],
                ["Germany", "83000000", "3800", "Europe"],
                ["Brazil", "210000000", "1800", "South America"],
            ],
        )
        return create_schema_chunk(table)

    def test_passes_through_non_table_chunks(self, step):
        """Test that non-table chunks pass through unchanged."""
        regular_chunk = Chunk(
            id="regular1",
            doc_id="doc1",
            content="This is regular content",
            chunk_index=0,
            metadata={},
        )

        result = step.execute("any query", [regular_chunk])

        assert len(result) == 1
        assert result[0].id == "regular1"
        assert result[0].content == "This is regular content"

    def test_processes_table_chunk(self, step, mock_chat_factory, sample_table_chunk):
        """Test processing a table schema chunk."""
        _, mock_chat = mock_chat_factory
        # Mock LLM responses
        mock_chat.chat.side_effect = [
            '["Country", "Population"]',  # Column selection
            'SELECT "country", "population" FROM "tbl_test123" WHERE "population"::NUMERIC > 100000000',  # SQL
        ]

        result = step.execute("Countries with population over 100 million", [sample_table_chunk])

        assert len(result) == 1
        # Should have augmented content with SQL results
        assert "SQL Query Results" in result[0].content
        assert "sql_executed" in result[0].metadata

    def test_handles_mixed_chunks(self, step, mock_chat_factory, sample_table_chunk):
        """Test handling mix of table and regular chunks."""
        _, mock_chat = mock_chat_factory
        regular_chunk = Chunk(
            id="reg1",
            doc_id="doc1",
            content="Regular content",
            chunk_index=0,
            metadata={},
        )

        # Mock for table chunk processing
        mock_chat.chat.side_effect = [
            '["Country"]',
            'SELECT "country" FROM "tbl_test123"',
        ]

        result = step.execute("query", [regular_chunk, sample_table_chunk])

        assert len(result) == 2
        # Regular chunks come first, then table results
        assert result[0].content == "Regular content"
        assert "SQL Query Results" in result[1].content

    def test_column_selection_fallback(self, step, mock_chat_factory, sample_table_chunk):
        """Test fallback when column selection returns invalid JSON."""
        _, mock_chat = mock_chat_factory
        mock_chat.chat.side_effect = [
            "invalid json response",  # Bad column selection
            'SELECT * FROM "tbl_test123"',  # SQL with all columns
        ]

        result = step.execute("query", [sample_table_chunk])

        # Should still work (fallback to all columns)
        assert len(result) == 1
        # Chat was called twice
        assert mock_chat.chat.call_count == 2

    def test_missing_table_in_store(self):
        """Test handling of chunk with table_id not in store."""
        factory, _ = create_mock_chat_factory()
        step = TableQueryStep(chat_factory=factory, table_store=MockTableStore())  # Empty store

        broken_chunk = Chunk(
            id="broken1",
            doc_id="doc1",
            content="Table schema",
            chunk_index=0,
            metadata={"is_table_schema": True, "table_id": "nonexistent"},
        )

        result = step.execute("query", [broken_chunk])

        # Should return original chunk on error
        assert len(result) == 1
        assert result[0].id == "broken1"

    def test_no_store_provided(self, sample_table_chunk):
        """Test handling when no table_store is provided."""
        factory, _ = create_mock_chat_factory()
        step = TableQueryStep(chat_factory=factory, table_store=None)

        result = step.execute("query", [sample_table_chunk])

        # Should return original chunk
        assert len(result) == 1
        assert result[0].id == sample_table_chunk.id


class TestColumnSelection:
    """Tests for column selection logic."""

    @pytest.fixture
    def step(self):
        factory, mock_chat = create_mock_chat_factory()
        return TableQueryStep(chat_factory=factory), mock_chat

    def test_parses_json_array(self, step):
        """Test parsing valid JSON array response."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = '["col1", "col2", "col3"]'

        columns = step_instance._select_columns("query", ["col1", "col2", "col3", "col4"])

        assert columns == ["col1", "col2", "col3"]

    def test_handles_markdown_code_block(self, step):
        """Test parsing JSON in markdown code block."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = """```json
["Country", "Population"]
```"""

        columns = step_instance._select_columns("query", ["Country", "Population", "GDP"])

        assert columns == ["Country", "Population"]

    def test_fallback_on_invalid_json(self, step):
        """Test fallback when JSON is invalid."""
        step_instance, mock_chat = step
        mock_chat.chat.return_value = "not valid json at all"

        all_columns = ["A", "B", "C"]
        columns = step_instance._select_columns("query", all_columns)

        # Should fallback to all columns
        assert columns == all_columns


class TestSQLExtraction:
    """Tests for SQL extraction logic."""

    def test_extracts_sql_from_plain_response(self):
        """Test extracting SQL from plain response."""
        response = "SELECT Country FROM data WHERE Population > 1000000"
        sql = extract_sql(response)

        assert sql.upper().startswith("SELECT")

    def test_extracts_sql_from_code_block(self):
        """Test extracting SQL from markdown code block."""
        response = """Here's the query:
```sql
SELECT Country, Population FROM data WHERE Population > 1000000
```
This will return countries with large populations."""

        sql = extract_sql(response)

        assert "SELECT" in sql.upper()
        assert "Population" in sql

    def test_extracts_sql_from_code_block_no_lang(self):
        """Test extracting SQL from code block without language tag."""
        response = """```
SELECT * FROM data
```"""

        sql = extract_sql(response)

        assert "SELECT" in sql.upper()


class TestResultFormatting:
    """Tests for result formatting."""

    @pytest.fixture
    def step(self):
        factory, _ = create_mock_chat_factory()
        return TableQueryStep(chat_factory=factory)

    def test_formats_results_as_markdown_table(self, step):
        """Test markdown table formatting."""
        result = step._format_as_markdown(
            ["Name", "Value"],
            [["Alice", 100], ["Bob", 200]],
        )

        assert "| Name | Value |" in result
        assert "| --- | --- |" in result
        assert "| Alice | 100 |" in result
        assert "| Bob | 200 |" in result

    def test_handles_empty_results(self, step):
        """Test formatting empty results."""
        result = step._format_as_markdown(["Col"], [])

        assert "(no results)" in result

    def test_truncates_long_values(self, step):
        """Test truncation of long cell values."""
        long_value = "A" * 100
        result = step._format_as_markdown(["Data"], [[long_value]])

        # Should be truncated
        assert "..." in result
        assert len(result) < len(long_value)

    def test_limits_result_rows(self, step):
        """Test limiting number of displayed rows."""
        step.max_results = 5
        many_rows = [[f"row{i}"] for i in range(100)]

        result = step._format_as_markdown(["ID"], many_rows)

        # Should mention more rows
        assert "more rows" in result


class TestMultiTableDetection:
    """Tests for multi-table query detection."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock store with multiple tables."""
        store = MockTableStore()
        store.add_table(
            table_id="emp123",
            pg_name="tbl_employees",
            sanitized_cols=["id", "name", "dept_id"],
            original_cols=["id", "name", "dept_id"],
            rows=[["1", "Alice", "10"], ["2", "Bob", "20"]],
        )
        store.add_table(
            table_id="dept456",
            pg_name="tbl_departments",
            sanitized_cols=["id", "dept_name", "location"],
            original_cols=["id", "dept_name", "location"],
            rows=[["10", "Engineering", "Building A"], ["20", "Sales", "Building B"]],
        )
        return store

    @pytest.fixture
    def employees_chunk(self):
        """Create employees table chunk."""
        table = ParsedTable(
            table_id="emp123",
            source_doc="employees.csv",
            headers=["id", "name", "dept_id"],
            rows=[["1", "Alice", "10"], ["2", "Bob", "20"], ["3", "Carol", "10"]],
        )
        return create_schema_chunk(table)

    @pytest.fixture
    def departments_chunk(self):
        """Create departments table chunk."""
        table = ParsedTable(
            table_id="dept456",
            source_doc="departments.csv",
            headers=["id", "dept_name", "location"],
            rows=[["10", "Engineering", "Building A"], ["20", "Sales", "Building B"]],
        )
        return create_schema_chunk(table)

    def test_returns_false_for_single_chunk(self, mock_store, employees_chunk):
        """Test detection returns false for single table."""
        factory, mock_chat = create_mock_chat_factory()
        step = TableQueryStep(chat_factory=factory, table_store=mock_store)

        result = step._needs_multi_table("query", [employees_chunk])

        assert result is False
        mock_chat.chat.assert_not_called()

    def test_calls_llm_for_multiple_chunks(self, mock_store, employees_chunk, departments_chunk):
        """Test detection calls LLM for multiple tables."""
        factory, mock_chat = create_mock_chat_factory()
        mock_chat.chat.return_value = "no"
        step = TableQueryStep(chat_factory=factory, table_store=mock_store)

        result = step._needs_multi_table(
            "What is Alice's name?",
            [employees_chunk, departments_chunk],
        )

        assert result is False
        mock_chat.chat.assert_called_once()

    def test_returns_true_for_join_query(self, mock_store, employees_chunk, departments_chunk):
        """Test detection returns true for join query."""
        factory, mock_chat = create_mock_chat_factory()
        mock_chat.chat.return_value = "yes"
        step = TableQueryStep(chat_factory=factory, table_store=mock_store)

        result = step._needs_multi_table(
            "Show employee names with their department names",
            [employees_chunk, departments_chunk],
        )

        assert result is True


class TestEndToEnd:
    """End-to-end tests for table query flow."""

    def test_full_query_flow(self):
        """Test complete flow from chunk to augmented result."""
        # Create table
        table = ParsedTable(
            table_id="e2e_test",
            source_doc="countries.csv",
            headers=["Country", "Population", "Capital"],
            rows=[
                ["USA", "330000000", "Washington"],
                ["France", "67000000", "Paris"],
                ["Japan", "125000000", "Tokyo"],
            ],
        )
        chunk = create_schema_chunk(table)

        # Setup mock store
        mock_store = MockTableStore()
        mock_store.add_table(
            table_id="e2e_test",
            pg_name="tbl_e2e_test",
            sanitized_cols=["country", "population", "capital"],
            original_cols=["Country", "Population", "Capital"],
            rows=[
                ["USA", "330000000", "Washington"],
                ["France", "67000000", "Paris"],
                ["Japan", "125000000", "Tokyo"],
            ],
        )

        # Setup mock chat factory
        factory, mock_chat = create_mock_chat_factory()
        mock_chat.chat.side_effect = [
            '["Country", "Capital"]',  # Column selection
            'SELECT "country", "capital" FROM "tbl_e2e_test"',  # SQL
        ]

        step = TableQueryStep(chat_factory=factory, table_store=mock_store)
        result = step.execute("What are the capitals?", [chunk])

        assert len(result) == 1
        # Check augmented content
        assert "SQL Query Results" in result[0].content
        # Results should include data
        assert "USA" in result[0].content or "Washington" in result[0].content
