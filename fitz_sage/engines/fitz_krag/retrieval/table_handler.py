# fitz_sage/engines/fitz_krag/retrieval/table_handler.py
"""
Table query handler — LLM SQL generation and execution for TABLE read results.

Runs after expansion, before assembly. Takes ReadResults that contain table
schemas and replaces their content with actual SQL query results.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from fitz_sage.core.identifiers import exact_identifiers
from fitz_sage.engines.fitz_krag.retrieval.table_plan import (
    build_table_query_plan,
    execute_table_query_plan,
)
from fitz_sage.engines.fitz_krag.types import AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.llm.providers.base import ChatProvider
    from fitz_sage.tabular.store.sqlite import SqliteTableStore

logger = logging.getLogger(__name__)

_SCAN_ROW_LIMIT = 500
SQL_PROMPT = """Generate a SQLite query to answer this question.

Table name: {table_name}
Columns (all TEXT type): {columns}
Sample data:
{samples}

Question: {question}

Rules:
1. Use only the columns listed above
2. Use the exact table name: {table_name}
3. Use LIMIT {max_results} unless aggregating
4. For text search use LIKE with '%pattern%' (SQLite LIKE is case-insensitive for ASCII)
5. For "highest/maximum" use ORDER BY column DESC LIMIT 1
6. For "lowest/minimum" use ORDER BY column ASC LIMIT 1
7. For "who/which" questions, include identifying columns (name, id) in SELECT
8. For numeric operations (MAX, MIN, AVG, SUM, ORDER BY numbers), \
use CAST(column AS REAL) or CAST(column AS INTEGER)
9. ALWAYS include in SELECT every column used in ORDER BY, WHERE, or GROUP BY
10. CRITICAL: When using COUNT, SUM, AVG with non-aggregated columns, \
you MUST add GROUP BY. Example: SELECT department, COUNT(*) FROM t GROUP BY department

Return ONLY the SQL query, no explanation."""


class TableQueryHandler:
    """Generates SQL and executes queries for TABLE read results."""

    def __init__(
        self,
        chat: "ChatProvider | None",
        sqlite_table_store: "SqliteTableStore",
        config: "FitzKragConfig",
    ):
        self._chat = chat
        self._sqlite_table_store = sqlite_table_store
        self._config = config

    def process(
        self,
        query: str,
        read_results: list[ReadResult],
        *,
        allow_sql_generation: bool = True,
    ) -> list[ReadResult]:
        """Identify TABLE results, generate SQL, execute, augment content."""
        table_results = [r for r in read_results if r.address.kind == AddressKind.TABLE]
        non_table_results = [r for r in read_results if r.address.kind != AddressKind.TABLE]

        if not table_results:
            return read_results

        augmented: list[ReadResult] = []
        for result in table_results:
            try:
                if self._chat is None or not allow_sql_generation:
                    logger.debug("Table SQL generation skipped; using deterministic row grounding")
                    aug = self._process_table_result_deterministic(query, result)
                else:
                    aug = self._process_table_result(query, result)
                augmented.append(aug)
            except Exception as e:
                logger.warning(f"Table query failed for {result.address.location}: {e}")
                augmented.append(result)

        return non_table_results + augmented

    def _process_table_result(self, query: str, result: ReadResult) -> ReadResult:
        """Process a single TABLE ReadResult: SQL gen → execute (single LLM call)."""
        table_id = result.metadata.get("table_id") or result.address.metadata.get("table_id")
        if not table_id:
            return result

        table_name = self._sqlite_table_store.get_table_name(table_id)
        if not table_name:
            return result

        col_info = self._sqlite_table_store.get_columns(table_id)
        if not col_info:
            return result

        sanitized_cols, original_cols = col_info
        row_count = self._sqlite_table_store.get_row_count(table_id)
        sample_rows = self._get_sample_data(table_name, sanitized_cols)

        # Generate SQL and get validated result in one pass (no double execution)
        sql, col_names, rows = self._generate_and_execute_sql(
            query, table_name, sanitized_cols, sample_rows
        )
        if not rows:
            return self._process_table_result_deterministic(query, result)

        name = result.address.metadata.get("name", result.address.location)
        columns = result.address.metadata.get("columns", original_cols)

        content = self._format_table_evidence(
            name=name,
            columns=columns,
            row_count=row_count,
            heading="SQL Query Results",
            query_label=f"Query: {sql}",
            col_names=col_names,
            rows=rows,
            note=f"Results computed from all {row_count} rows.",
        )

        return ReadResult(
            address=result.address,
            content=content,
            file_path=result.file_path,
            metadata={**result.metadata, "sql_executed": sql, "result_count": len(rows)},
        )

    def _process_table_result_deterministic(self, query: str, result: ReadResult) -> ReadResult:
        """Return query-grounded table evidence without relying on generated SQL."""
        table_id = result.metadata.get("table_id") or result.address.metadata.get("table_id")
        if not table_id:
            return result

        table_name = self._sqlite_table_store.get_table_name(table_id)
        if not table_name:
            return result

        col_info = self._sqlite_table_store.get_columns(table_id)
        if not col_info:
            return result

        sanitized_cols, original_cols = col_info
        row_count = self._sqlite_table_store.get_row_count(table_id)
        identifiers = tuple(exact_identifiers(query))
        row_columns = sanitized_cols
        rows: list[list[Any]] = []
        exact_identifier_lookup = False
        if identifiers:
            find_rows = getattr(self._sqlite_table_store, "find_rows_by_identifiers", None)
            if callable(find_rows):
                candidate = find_rows(
                    table_id,
                    identifiers,
                    limit=max(self._config.max_table_results, len(identifiers)),
                )
                if _is_row_data(candidate):
                    row_columns, rows = candidate
                    exact_identifier_lookup = True
        if not exact_identifier_lookup:
            rows = self._get_scan_data(table_name, sanitized_cols)

        plan = build_table_query_plan(query, row_columns, rows)
        selected_rows = execute_table_query_plan(plan, rows)
        if not selected_rows:
            return result

        name = result.address.metadata.get("name", result.address.location)
        columns = result.address.metadata.get("columns", original_cols)
        result_columns = original_cols if len(original_cols) == len(row_columns) else row_columns
        lookup_note = (
            f"Rows selected by exact identifier lookup across all {row_count} row(s)."
            if exact_identifier_lookup
            else (
                f"Rows selected from a bounded scan of {len(rows)} row(s)"
                f" out of {row_count} total row(s)."
            )
        )
        content = self._format_table_evidence(
            name=name,
            columns=columns,
            row_count=row_count,
            heading="Deterministic Table Matches",
            query_label="Selection: query-grounded row filter",
            col_names=result_columns,
            rows=selected_rows,
            note=lookup_note,
        )

        return ReadResult(
            address=result.address,
            content=content,
            file_path=result.file_path,
            metadata={
                **result.metadata,
                "deterministic_table_filter": True,
                "exact_identifier_table_lookup": exact_identifier_lookup,
                "result_count": len(selected_rows),
                "table_query_plan": plan.metadata,
            },
        )

    def _get_sample_data(
        self, table_name: str, columns: list[str], limit: int = 3
    ) -> list[list[str]]:
        """Fetch sample data from SQLite table."""
        cols_str = ", ".join(f'"{c}"' for c in columns)
        sql = f'SELECT {cols_str} FROM "{table_name}" LIMIT {limit}'
        result = self._sqlite_table_store.execute_query("", sql)
        if result:
            _, rows = result
            return [[str(v) if v is not None else "" for v in row] for row in rows]
        return []

    def _get_scan_data(self, table_name: str, columns: list[str]) -> list[list[Any]]:
        """Fetch a bounded row scan for deterministic table grounding."""
        limit = max(
            self._config.max_table_results,
            min(_SCAN_ROW_LIMIT, self._config.max_table_results * 10),
        )
        cols_str = ", ".join(f'"{c}"' for c in columns)
        sql = f'SELECT {cols_str} FROM "{table_name}" ORDER BY _row_num LIMIT {limit}'
        result = self._sqlite_table_store.execute_query("", sql)
        if result:
            _, rows = result
            return rows
        return []

    def _generate_and_execute_sql(
        self,
        query: str,
        table_name: str,
        columns: list[str],
        sample_rows: list[list[str]],
        max_retries: int = 2,
    ) -> tuple[str, list[str], list[list]]:
        """Generate SQL, validate by execution, and return the result.

        Returns (sql, col_names, rows). On total failure returns (sql, [], []).
        Reuses the successful validation execution — no double round-trip.
        """
        previous_error = None

        for attempt in range(max_retries + 1):
            sql = self._generate_sql_attempt(
                query, table_name, columns, sample_rows, previous_error
            )

            result = self._sqlite_table_store.execute_query("", sql)
            if result is not None:
                col_names, rows = result
                return sql, col_names, rows

            # Capture actual SQLite error for the retry prompt
            previous_error = self._capture_sql_error(sql)
            logger.warning(f"SQL validation failed (attempt {attempt + 1}/{max_retries + 1})")

        return sql, [], []

    def _capture_sql_error(self, sql: str) -> str:
        """Re-execute SQL to capture the SQLite error message."""
        from fitz_sage.storage import get_connection_manager

        try:
            cm = get_connection_manager()
            with cm.connection(self._sqlite_table_store.collection) as conn:
                conn.execute(sql.replace("%", "%%"))
            return "Query execution failed"
        except Exception as e:
            return str(e)

    def _generate_sql_attempt(
        self,
        query: str,
        table_name: str,
        columns: list[str],
        sample_rows: list[list[str]],
        previous_error: str | None,
    ) -> str:
        """Generate SQL query (single attempt)."""
        # Format samples as readable rows (col=val pairs)
        sample_lines = []
        for row in sample_rows[:3]:
            pairs = [f"{col}={val}" for col, val in zip(columns, row) if val]
            sample_lines.append(" | ".join(pairs[:10]))
        samples_str = "\n".join(sample_lines) if sample_lines else "(no sample data)"

        error_context = ""
        if previous_error:
            error_context = (
                f"\nIMPORTANT: Your previous SQL failed with:\n"
                f"{previous_error}\n"
                f"Fix this error in your new query."
            )

        prompt = SQL_PROMPT.format(
            table_name=table_name,
            columns=columns,
            samples=samples_str,
            question=query,
            max_results=self._config.max_table_results,
        )
        prompt = prompt + error_context

        if self._chat is None:
            raise RuntimeError("SQL generation requires a configured chat provider")
        response = self._chat.chat([{"role": "user", "content": prompt}])
        return self._extract_sql(response)

    def _format_as_markdown(self, cols: list[str], rows: list[list[Any]]) -> str:
        """Format query results as markdown table."""
        if not rows:
            return "(no results)"

        max_results = self._config.max_table_results
        display_rows = rows[:max_results]

        lines = []
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")
        lines.append("| " + " | ".join("---" for _ in cols) + " |")

        for row in display_rows:
            cells = []
            for val in row:
                s = str(val) if val is not None else ""
                if len(s) > 50:
                    s = s[:47] + "..."
                cells.append(s)
            lines.append("| " + " | ".join(cells) + " |")

        if len(rows) > max_results:
            lines.append(f"\n... and {len(rows) - max_results} more rows")

        return "\n".join(lines)

    def _format_table_evidence(
        self,
        *,
        name: str,
        columns: list[str],
        row_count: int | None,
        heading: str,
        query_label: str,
        col_names: list[str],
        rows: list[list[Any]],
        note: str,
    ) -> str:
        """Format table evidence with a stable provenance envelope."""
        results_md = self._format_as_markdown(col_names, rows)
        total_rows = row_count if row_count is not None else "unknown"
        return (
            f"Table: {name}\n"
            f"Columns: {', '.join(columns)}\n"
            f"Total rows: {total_rows}\n\n"
            f"--- {heading} ---\n"
            f"{query_label}\n"
            f"Results ({len(rows)} rows):\n"
            f"{results_md}\n\n"
            f"Note: {note}"
        )

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from LLM response."""
        text = response.strip()

        if "```" in text:
            match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1).strip()
            else:
                text = text.replace("```sql", "").replace("```", "").strip()

        if not text.upper().startswith("SELECT"):
            match = re.search(r"(SELECT\s+.+)", text, re.DOTALL | re.IGNORECASE)
            if match:
                text = match.group(1)

        return text


def _is_row_data(value: Any) -> bool:
    """Return whether a store result has the expected columns-and-rows shape."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], list)
        and isinstance(value[1], list)
    )
