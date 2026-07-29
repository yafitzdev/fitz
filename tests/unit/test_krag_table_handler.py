# tests/unit/test_krag_table_handler.py
"""Tests for TableQueryHandler (query-time SQL generation)."""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.query_pipeline import _closure_result_should_replace
from fitz_sage.engines.fitz_krag.retrieval.table_handler import TableQueryHandler
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(
    chat_response: str | list[str] = 'SELECT "col1", "col2" FROM "tbl_sales" LIMIT 100',
    execute_result: tuple | None = (["col1", "col2"], [["a", "b"], ["c", "d"]]),
    table_name: str = "tbl_sales",
    columns: tuple | None = (["col1", "col2"], ["Col 1", "Col 2"]),
    row_count: int = 100,
    max_table_results: int = 100,
) -> tuple[TableQueryHandler, MagicMock, MagicMock]:
    chat = MagicMock(name="chat")
    if isinstance(chat_response, list):
        chat.chat.side_effect = chat_response
    else:
        chat.chat.return_value = chat_response

    sqlite_table_store = MagicMock(name="sqlite_table_store")
    sqlite_table_store.get_table_name.return_value = table_name
    sqlite_table_store.get_columns.return_value = columns
    sqlite_table_store.get_row_count.return_value = row_count
    sqlite_table_store.execute_query.return_value = execute_result

    config = MagicMock(name="config")
    config.max_table_results = max_table_results

    handler = TableQueryHandler(chat, sqlite_table_store, config)
    return handler, chat, sqlite_table_store


def _make_table_read_result(
    table_id: str = "tbl_abc",
    table_index_id: str = "rec-001",
    name: str = "Sales Data",
    columns: list[str] | None = None,
    row_count: int = 100,
) -> ReadResult:
    addr = Address(
        kind=AddressKind.TABLE,
        source_id="file1",
        location=name,
        summary=f"Table {name}",
        score=0.9,
        metadata={
            "table_index_id": table_index_id,
            "table_id": table_id,
            "name": name,
            "columns": columns or ["col1", "col2"],
            "row_count": row_count,
        },
    )
    return ReadResult(
        address=addr,
        content=f"Table: {name}\nColumns: col1, col2\nRow count: {row_count}",
        file_path="data.csv",
        metadata={"table_id": table_id},
    )


def _make_non_table_result() -> ReadResult:
    addr = Address(
        kind=AddressKind.SYMBOL,
        source_id="file1",
        location="mod.func",
        summary="A function",
        score=0.8,
        metadata={"start_line": 1, "end_line": 5},
    )
    return ReadResult(
        address=addr,
        content="def func(): pass",
        file_path="module.py",
        line_range=(1, 5),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTableQueryHandler:
    def test_process_skips_non_table(self):
        """Non-TABLE ReadResults pass through unchanged."""
        handler, _, _ = _make_handler()
        non_table = _make_non_table_result()

        results = handler.process("query", [non_table])

        assert len(results) == 1
        assert results[0] is non_table

    def test_process_generates_sql(self):
        """TABLE result triggers SQL generation + execution."""
        handler, chat, pg_store = _make_handler(
            chat_response='SELECT "col1", "col2" FROM "tbl_sales" LIMIT 100'
        )
        table_result = _make_table_read_result()

        results = handler.process("what are the top sales?", [table_result])

        assert len(results) == 1
        # Chat called once (combined SQL generation)
        assert chat.chat.call_count >= 1
        # execute_query called for sample data + validation + final execution
        assert pg_store.execute_query.call_count >= 1

    def test_process_augments_content(self):
        """Content is replaced with SQL results."""
        handler, _, _ = _make_handler(chat_response='SELECT "col1" FROM "tbl_sales" LIMIT 100')
        table_result = _make_table_read_result()

        results = handler.process("query", [table_result])

        assert len(results) == 1
        content = results[0].content
        assert "SQL Query Results" in content
        assert "Sales Data" in content
        assert "col1" in content

    def test_process_mixed_results(self):
        """Mix of TABLE and non-TABLE results handled correctly."""
        handler, _, _ = _make_handler(chat_response='SELECT "col1" FROM "tbl_sales"')

        non_table = _make_non_table_result()
        table_result = _make_table_read_result()

        results = handler.process("query", [non_table, table_result])

        assert len(results) == 2
        # Non-table comes first
        assert results[0].address.kind == AddressKind.SYMBOL
        # Table result augmented
        assert results[1].address.kind == AddressKind.TABLE
        assert "SQL Query Results" in results[1].content

    def test_process_no_tables(self):
        """Returns input unchanged when no TABLE results."""
        handler, chat, _ = _make_handler()
        non_table = _make_non_table_result()

        results = handler.process("query", [non_table])

        assert results == [non_table]
        chat.chat.assert_not_called()

    def test_process_graceful_on_missing_table(self):
        """Falls back to original when table not found in store."""
        handler, _, pg_store = _make_handler()
        pg_store.get_table_name.return_value = None  # Table not found
        table_result = _make_table_read_result()

        results = handler.process("query", [table_result])

        assert len(results) == 1
        # Original content preserved
        assert "SQL Query Results" not in results[0].content

    def test_process_execution_failure(self):
        """Falls back to original when SQL execution fails."""
        responses = ['["col1"]', "SELECT bad_sql"]
        handler, _, pg_store = _make_handler(chat_response=responses)
        # All execute_query calls return None (failure)
        pg_store.execute_query.return_value = None
        table_result = _make_table_read_result()

        results = handler.process("query", [table_result])

        assert len(results) == 1
        # Original result returned on failure
        assert results[0].address.kind == AddressKind.TABLE

    def test_process_without_chat_handles_prefixed_boolean_negation(self):
        """An un-prefixed column term should select false boolean rows."""
        sqlite_table_store = MagicMock(name="sqlite_table_store")
        sqlite_table_store.get_table_name.return_value = "tbl_builds"
        sqlite_table_store.get_columns.return_value = (
            ["build_id", "environment", "approved_for_release"],
            ["build_id", "environment", "approved_for_release"],
        )
        sqlite_table_store.get_row_count.return_value = 3
        sqlite_table_store.execute_query.return_value = (
            ["build_id", "environment", "approved_for_release"],
            [
                ["BLD-22", "Prod", "no"],
                ["BLD-23", "Prod", "yes"],
                ["BLD-24", "Stage", "no"],
            ],
        )
        config = MagicMock(name="config")
        config.max_table_results = 100
        handler = TableQueryHandler(None, sqlite_table_store, config)
        table_result = _make_table_read_result(name="Builds")

        results = handler.process(
            "Which Prod builds are unapproved for release?",
            [table_result],
        )

        assert "Deterministic Table Matches" in results[0].content
        assert "BLD-22" in results[0].content
        assert "BLD-23" not in results[0].content
        assert "BLD-24" not in results[0].content
        assert results[0].metadata["deterministic_table_filter"] is True
        assert results[0].metadata["table_query_plan"]["predicates"] == [
            {
                "column": "environment",
                "accepted_values": ["prod"],
                "source": "categorical_value",
            },
            {
                "column": "approved_for_release",
                "accepted_values": ["0", "disabled", "false", "inactive", "no", "off"],
                "source": "boolean_column",
            },
        ]

    def test_deterministic_table_filter_scopes_superlative_to_row_values(self):
        """Superlative row selection should filter to entity-matching rows before sorting."""
        sqlite_table_store = MagicMock(name="sqlite_table_store")
        sqlite_table_store.get_table_name.return_value = "tbl_experiments"
        sqlite_table_store.get_columns.return_value = (
            ["experiment_id", "project", "conversion_rate"],
            ["experiment_id", "project", "conversion_rate"],
        )
        sqlite_table_store.get_row_count.return_value = 3
        sqlite_table_store.execute_query.return_value = (
            ["experiment_id", "project", "conversion_rate"],
            [
                ["EXP-21", "Atlas", "6.2"],
                ["EXP-22", "Atlas", "7.4"],
                ["EXP-99", "Zephyr", "9.9"],
            ],
        )
        config = MagicMock(name="config")
        config.max_table_results = 100
        handler = TableQueryHandler(None, sqlite_table_store, config)
        table_result = _make_table_read_result(name="Experiments")

        results = handler.process(
            "Which Atlas experiment had the highest conversion rate?",
            [table_result],
        )

        assert "EXP-22" in results[0].content
        assert "EXP-99" not in results[0].content

    def test_process_can_disable_sql_generation_for_evidence_mode(self):
        """Evidence mode can ground table rows without calling the chat SQL generator."""
        handler, chat, sqlite_table_store = _make_handler(
            execute_result=(
                ["invoice_id", "vendor", "late_fee_percent"],
                [["INV-702", "Cobalt CDN", "2.0"], ["INV-703", "Skyline Labs", "0.0"]],
            ),
            table_name="tbl_invoices",
            columns=(
                ["invoice_id", "vendor", "late_fee_percent"],
                ["invoice_id", "vendor", "late_fee_percent"],
            ),
            row_count=2,
        )
        table_result = _make_table_read_result(name="Invoices")

        results = handler.process(
            "Which vendor owns INV-702?",
            [table_result],
            allow_sql_generation=False,
        )

        chat.chat.assert_not_called()
        assert "Deterministic Table Matches" in results[0].content
        assert "Cobalt CDN" in results[0].content

    def test_deterministic_table_filter_keeps_exact_identifier_for_boolean_question(self):
        """Exact row lookup should not discard a row because the boolean answer is false."""
        sqlite_table_store = MagicMock(name="sqlite_table_store")
        sqlite_table_store.get_table_name.return_value = "tbl_assets"
        sqlite_table_store.get_columns.return_value = (
            ["asset_id", "encrypted", "owner"],
            ["asset_id", "encrypted", "owner"],
        )
        sqlite_table_store.get_row_count.return_value = 2
        sqlite_table_store.execute_query.return_value = (
            ["asset_id", "encrypted", "owner"],
            [["AST-22", "no", "Data Platform"], ["AST-31", "yes", "Search Platform"]],
        )
        config = MagicMock(name="config")
        config.max_table_results = 100
        handler = TableQueryHandler(None, sqlite_table_store, config)
        table_result = _make_table_read_result(name="Assets")

        results = handler.process("Is asset AST-22 encrypted?", [table_result])

        assert "AST-22" in results[0].content
        assert "Data Platform" in results[0].content
        assert "AST-31" not in results[0].content

    def test_deterministic_table_filter_uses_full_table_identifier_lookup(self):
        """Exact identifier evidence should not depend on the bounded scan prefix."""
        handler, _, sqlite_table_store = _make_handler(
            execute_result=None,
            table_name="tbl_assets",
            columns=(
                ["asset_id", "owner"],
                ["asset_id", "owner"],
            ),
            row_count=900,
        )
        sqlite_table_store.find_rows_by_identifiers.return_value = (
            ["asset_id", "owner"],
            [["AX-156", "Platform"]],
        )
        table_result = _make_table_read_result(name="Assets", columns=["asset_id", "owner"])

        results = handler.process(
            "Who owns AX-156?",
            [table_result],
            allow_sql_generation=False,
        )

        assert "AX-156" in results[0].content
        assert "Platform" in results[0].content
        assert results[0].metadata["exact_identifier_table_lookup"] is True
        sqlite_table_store.execute_query.assert_not_called()

    def test_deterministic_table_filter_uses_bm25_rows_beyond_scan_window(self):
        """Free-text row retrieval should preserve its concrete distant row."""
        handler, _, sqlite_table_store = _make_handler(
            execute_result=None,
            table_name="tbl_assets",
            columns=(
                ["asset_name", "owner"],
                ["asset_name", "owner"],
            ),
            row_count=900,
        )
        sqlite_table_store.get_rows_by_numbers.return_value = (
            ["asset_name", "owner"],
            [["Meridian relay", "Platform"]],
        )
        table_result = _make_table_read_result(
            name="Assets",
            columns=["asset_name", "owner"],
        )
        table_result.address.metadata["row_search"] = {"row_numbers": [640]}

        results = handler.process(
            "Who owns the Meridian relay?",
            [table_result],
            allow_sql_generation=False,
        )

        assert "Meridian relay" in results[0].content
        assert "Platform" in results[0].content
        assert results[0].metadata["indexed_table_row_lookup"] is True
        sqlite_table_store.get_rows_by_numbers.assert_called_once_with(
            "tbl_abc",
            [640],
            limit=100,
        )
        sqlite_table_store.execute_query.assert_not_called()

    def test_noisier_closure_table_result_does_not_replace_precise_result(self):
        """Closure provenance alone must not replace a narrower table result."""
        existing = _make_table_read_result(name="Assets")
        existing.metadata.update(
            {
                "deterministic_table_filter": True,
                "result_count": 1,
            }
        )
        candidate = _make_table_read_result(name="Assets")
        candidate.metadata.update(
            {
                "deterministic_table_filter": True,
                "evidence_closure": {"role": "bridge:ASSET-940"},
                "result_count": 4,
            }
        )

        assert _closure_result_should_replace(existing, candidate) is False

    def test_deterministic_table_plan_does_not_sort_on_identifier_columns(self):
        """Metric superlatives should bind to metric columns, not numeric-looking IDs."""
        sqlite_table_store = MagicMock(name="sqlite_table_store")
        sqlite_table_store.get_table_name.return_value = "tbl_exports"
        sqlite_table_store.get_columns.return_value = (
            ["export_id", "dataset", "rows"],
            ["export_id", "dataset", "rows"],
        )
        sqlite_table_store.get_row_count.return_value = 3
        sqlite_table_store.execute_query.return_value = (
            ["export_id", "dataset", "rows"],
            [
                ["EXP-501", "customer_export_logs", "120000"],
                ["EXP-504", "audit_trail", "990000"],
                ["EXP-505", "biometric_artifacts", "1200"],
            ],
        )
        config = MagicMock(name="config")
        config.max_table_results = 100
        handler = TableQueryHandler(None, sqlite_table_store, config)
        table_result = _make_table_read_result(name="Exports")

        results = handler.process("Which export has the most rows?", [table_result])

        assert "EXP-504" in results[0].content
        assert "audit_trail" in results[0].content
        assert "EXP-501" not in results[0].content
        assert results[0].metadata["table_query_plan"]["sort"] == {
            "column": "rows",
            "direction": "max",
        }

    def test_deterministic_table_plan_uses_compound_boolean_column_before_superlative(self):
        """Compound boolean columns should scope the candidate set before ordering."""
        sqlite_table_store = MagicMock(name="sqlite_table_store")
        sqlite_table_store.get_table_name.return_value = "tbl_jobs"
        sqlite_table_store.get_columns.return_value = (
            ["job_id", "duration_minutes", "manual_review_enabled"],
            ["job_id", "duration_minutes", "manual_review_enabled"],
        )
        sqlite_table_store.get_row_count.return_value = 3
        sqlite_table_store.execute_query.return_value = (
            ["job_id", "duration_minutes", "manual_review_enabled"],
            [
                ["JOB-611", "42", "yes"],
                ["JOB-724", "26", "no"],
                ["JOB-744", "88", "no"],
            ],
        )
        config = MagicMock(name="config")
        config.max_table_results = 100
        handler = TableQueryHandler(None, sqlite_table_store, config)
        table_result = _make_table_read_result(name="Jobs")

        results = handler.process(
            "Which manual-review-enabled job has the longest duration?",
            [table_result],
        )

        assert "JOB-611" in results[0].content
        assert "JOB-744" not in results[0].content
        predicates = results[0].metadata["table_query_plan"]["predicates"]
        assert predicates == [
            {
                "column": "manual_review_enabled",
                "accepted_values": ["1", "active", "enabled", "on", "true", "yes"],
                "source": "boolean_column",
            }
        ]

    def test_deterministic_table_plan_executes_over_rows_beyond_scan_limit(self):
        """A typed superlative plan should execute over the complete stored table."""
        sqlite_table_store = MagicMock(name="sqlite_table_store")
        sqlite_table_store.get_table_name.return_value = "tbl_jobs"
        sqlite_table_store.get_columns.return_value = (
            ["job_id", "duration_minutes", "manual_review_enabled"],
            ["job_id", "duration_minutes", "manual_review_enabled"],
        )
        sqlite_table_store.get_row_count.return_value = 900
        sqlite_table_store.execute_query.return_value = (
            ["job_id", "duration_minutes", "manual_review_enabled"],
            [
                ["JOB-001", "42", "yes"],
                ["JOB-002", "88", "no"],
            ],
        )
        sqlite_table_store.select_rows.return_value = (
            ["job_id", "duration_minutes", "manual_review_enabled"],
            [["JOB-850", "999", "yes"]],
        )
        config = MagicMock(name="config")
        config.max_table_results = 100
        handler = TableQueryHandler(None, sqlite_table_store, config)
        table_result = _make_table_read_result(name="Jobs", row_count=900)

        results = handler.process(
            "Which manual-review-enabled job has the longest duration?",
            [table_result],
        )

        assert "JOB-850" in results[0].content
        assert "JOB-001" not in results[0].content
        assert "across all 900 row(s)" in results[0].content
        assert results[0].metadata["full_table_plan_execution"] is True
        sqlite_table_store.select_rows.assert_called_once_with(
            "tbl_abc",
            predicates=(
                (
                    2,
                    ("1", "active", "enabled", "on", "true", "yes"),
                ),
            ),
            sort=(1, "max"),
            limit=1,
        )

    def test_deterministic_table_plan_binds_negation_to_its_boolean_column(self):
        """One negated condition must not invert another boolean condition."""
        sqlite_table_store = MagicMock(name="sqlite_table_store")
        sqlite_table_store.get_table_name.return_value = "tbl_services"
        sqlite_table_store.get_columns.return_value = (
            ["service", "active", "managed"],
            ["service", "active", "managed"],
        )
        sqlite_table_store.get_row_count.return_value = 3
        sqlite_table_store.execute_query.return_value = (
            ["service", "active", "managed"],
            [
                ["api", "yes", "no"],
                ["worker", "no", "no"],
                ["web", "yes", "yes"],
            ],
        )
        config = MagicMock(name="config")
        config.max_table_results = 100
        handler = TableQueryHandler(None, sqlite_table_store, config)
        table_result = _make_table_read_result(name="Services")

        results = handler.process(
            "Which active service is not managed?",
            [table_result],
        )

        assert "| api | yes | no |" in results[0].content
        assert "| worker |" not in results[0].content
        assert "| web |" not in results[0].content
        assert results[0].metadata["table_query_plan"]["predicates"] == [
            {
                "column": "active",
                "accepted_values": ["1", "active", "enabled", "on", "true", "yes"],
                "source": "boolean_column",
            },
            {
                "column": "managed",
                "accepted_values": ["0", "disabled", "false", "inactive", "no", "off"],
                "source": "boolean_column",
            },
        ]


class TestHelperMethods:
    def test_format_as_markdown_empty(self):
        handler, _, _ = _make_handler()
        assert handler._format_as_markdown(["a"], []) == "(no results)"

    def test_format_as_markdown_rows(self):
        handler, _, _ = _make_handler()
        result = handler._format_as_markdown(["name", "age"], [["Alice", "30"]])
        assert "| name | age |" in result
        assert "| Alice | 30 |" in result

    def test_extract_sql_plain(self):
        handler, _, _ = _make_handler()
        assert handler._extract_sql("SELECT * FROM t") == "SELECT * FROM t"

    def test_extract_sql_from_code_block(self):
        handler, _, _ = _make_handler()
        result = handler._extract_sql("```sql\nSELECT * FROM t\n```")
        assert result == "SELECT * FROM t"
