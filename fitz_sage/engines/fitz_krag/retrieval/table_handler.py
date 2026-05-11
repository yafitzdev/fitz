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

from fitz_sage.engines.fitz_krag.types import AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.llm.providers.base import ChatProvider
    from fitz_sage.tabular.store.sqlite import SqliteTableStore

logger = logging.getLogger(__name__)

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
        chat: "ChatProvider",
        pg_table_store: "SqliteTableStore",
        config: "FitzKragConfig",
    ):
        self._chat = chat
        self._pg_table_store = pg_table_store
        self._config = config

    def process(self, query: str, read_results: list[ReadResult]) -> list[ReadResult]:
        """Identify TABLE results, generate SQL, execute, augment content."""
        table_results = [r for r in read_results if r.address.kind == AddressKind.TABLE]
        non_table_results = [r for r in read_results if r.address.kind != AddressKind.TABLE]

        if not table_results:
            return read_results

        augmented: list[ReadResult] = []
        for result in table_results:
            try:
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

        pg_table_name = self._pg_table_store.get_table_name(table_id)
        if not pg_table_name:
            return result

        col_info = self._pg_table_store.get_columns(table_id)
        if not col_info:
            return result

        sanitized_cols, original_cols = col_info
        row_count = self._pg_table_store.get_row_count(table_id)
        sample_rows = self._get_sample_data(pg_table_name, sanitized_cols)

        # Generate SQL and get validated result in one pass (no double execution)
        sql, col_names, rows = self._generate_and_execute_sql(
            query, pg_table_name, sanitized_cols, sample_rows
        )
        if not rows:
            return result

        name = result.address.metadata.get("name", result.address.location)
        columns = result.address.metadata.get("columns", original_cols)

        results_md = self._format_as_markdown(col_names, rows)
        content = (
            f"Table: {name}\n"
            f"Columns: {', '.join(columns)}\n"
            f"Total rows: {row_count}\n\n"
            f"--- SQL Query Results ---\n"
            f"Query: {sql}\n"
            f"Results ({len(rows)} rows):\n"
            f"{results_md}\n\n"
            f"Note: Results computed from all {row_count} rows."
        )

        return ReadResult(
            address=result.address,
            content=content,
            file_path=result.file_path,
            metadata={**result.metadata, "sql_executed": sql, "result_count": len(rows)},
        )

    def _get_sample_data(
        self, pg_table_name: str, columns: list[str], limit: int = 3
    ) -> list[list[str]]:
        """Fetch sample data from PostgreSQL table."""
        cols_str = ", ".join(f'"{c}"' for c in columns)
        sql = f'SELECT {cols_str} FROM "{pg_table_name}" LIMIT {limit}'
        result = self._pg_table_store.execute_query("", sql)
        if result:
            _, rows = result
            return [[str(v) if v is not None else "" for v in row] for row in rows]
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

            result = self._pg_table_store.execute_query("", sql)
            if result is not None:
                col_names, rows = result
                return sql, col_names, rows

            # Capture actual PostgreSQL error for the retry prompt
            previous_error = self._capture_sql_error(sql)
            logger.warning(f"SQL validation failed (attempt {attempt + 1}/{max_retries + 1})")

        return sql, [], []

    def _capture_sql_error(self, sql: str) -> str:
        """Re-execute SQL to capture the PostgreSQL error message."""
        from fitz_sage.storage import get_connection_manager

        try:
            cm = get_connection_manager()
            with cm.connection(self._pg_table_store.collection) as conn:
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
