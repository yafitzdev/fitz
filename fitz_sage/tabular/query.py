# fitz_sage/tabular/query.py
"""
Table Query Step - Handles schema chunks at query time.

When a schema chunk is retrieved:
1. LLM selects relevant columns (column pruning)
2. LLM generates SQL query
3. Query executed directly against SQLite
4. Results are formatted and chunk is augmented

Multi-table join support:
- When multiple schema chunks are retrieved, LLM detects if query needs joins
- LLM generates SQL with JOINs using actual SQLite table names
- Query executed against SQLite database

Table storage:
- All tables stored as native SQLite tables via SqliteTableStore
- Direct SQL execution without data loading
- Efficient handling of large tables
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from fitz_sage.core.chunk import Chunk
from fitz_sage.core.json_utils import parse_llm_json
from fitz_sage.llm.factory import ChatFactory, ModelTier
from fitz_sage.tabular.sql_gen import extract_sql

from .store.sqlite import SqliteTableStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TableQueryStep:
    """
    Retrieval step that handles table schema chunks.

    For each schema chunk retrieved:
    1. Gets table metadata from SqliteTableStore
    2. LLM selects relevant columns (column pruning)
    3. LLM generates SQL query
    4. Query executed directly against SQLite
    5. Results formatted and chunk augmented

    Regular (non-table) chunks pass through unchanged.

    Args:
        chat_factory: Chat factory for per-task tier selection.
        max_results: Maximum number of SQL result rows to include (default: 100).
        table_store: SqliteTableStore for executing queries.
    """

    chat_factory: ChatFactory  # Chat factory for LLM calls
    max_results: int = 100
    table_store: "SqliteTableStore | None" = field(default=None)

    # Tier for table query tasks (developer decision - balanced for SQL generation)
    TIER_COLUMN_SELECT: ModelTier = "fast"
    TIER_SQL_GENERATE: ModelTier = "balanced"

    COLUMN_SELECT_PROMPT = """Given this table schema and question, which columns are needed to answer?

Table columns: {columns}
Sample data: {samples}
Question: {question}

Return ONLY a JSON array of column names needed. Example: ["col1", "col2"]
Rules:
- Include columns needed for filtering AND for displaying results
- Look at sample data to find which columns contain relevant values for the question
- For "who/which/what" questions, ALWAYS include identifying columns (name, id, title, etc.)
- For numeric comparisons (highest, lowest, average), include the numeric column
- When in doubt, include more columns rather than fewer"""

    SQL_PROMPT = """Generate a SQLite query to answer this question.

Table name: {table_name}
Columns: {columns}
Sample values: {samples}

Question: {question}

Rules:
1. Use only the columns listed above
2. Use the exact table name: {table_name}
3. Use LIMIT {max_results} unless aggregating
4. For text search use LIKE with % wildcards (SQLite LIKE is case-insensitive for ASCII)
5. Column names need double quotes if they contain special characters
6. For "highest/maximum" use ORDER BY column DESC LIMIT 1
7. For "lowest/minimum" use ORDER BY column ASC LIMIT 1
8. For "who/which" questions, include identifying columns (name, id) in SELECT
9. All columns are TEXT type. For numeric operations (MAX, MIN, AVG, SUM, ORDER BY numbers), use CAST(column AS REAL) or CAST(column AS INTEGER)

Return ONLY the SQL query, no explanation."""

    MULTI_TABLE_DETECT_PROMPT = """Does answering this question require combining data from multiple tables?

Available tables and their columns:
{table_schemas}

Question: {question}

Answer ONLY "yes" or "no"."""

    MULTI_TABLE_SQL_PROMPT = """Generate a SQLite query to answer this question using these tables.

Tables:
{table_schemas}

Question: {question}

Rules:
1. Use the exact table names shown above
2. If tables need to be joined, infer join columns from matching column names
3. Use LIMIT {max_results} unless aggregating
4. Column names need double quotes if they contain special characters
5. Use LIKE for case-insensitive text search (SQLite LIKE is case-insensitive for ASCII)
6. All columns are TEXT type - CAST to REAL or INTEGER for numeric operations

Return ONLY the SQL query, no explanation."""

    def execute(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Process chunks, with multi-table join support.

        When multiple schema chunks are retrieved:
        1. LLM decides if query needs multiple tables
        2. If yes, generate JOIN SQL executed against SQLite
        3. If no, each table processed independently

        Args:
            query: User query.
            chunks: Retrieved chunks (may include schema chunks).

        Returns:
            Chunks with table schema chunks augmented with query results.
        """
        table_chunks = [c for c in chunks if c.metadata.get("is_table_schema")]
        regular_chunks = [c for c in chunks if not c.metadata.get("is_table_schema")]

        if not table_chunks:
            return chunks

        # Check if multi-table query needed
        if self._needs_multi_table(query, table_chunks):
            logger.debug(f"Multi-table query detected for {len(table_chunks)} tables")
            result_chunk = self._process_multi_table(query, table_chunks)
            return regular_chunks + [result_chunk]

        # Single-table processing
        result_chunks: list[Chunk] = []
        for chunk in table_chunks:
            augmented = self._process_table_chunk(query, chunk)
            result_chunks.append(augmented)

        return regular_chunks + result_chunks

    def _get_table_info(self, chunk: Chunk) -> tuple[str, list[str], list[str]] | None:
        """
        Get table info from chunk metadata and store.

        Returns:
            Tuple of (table_name, sanitized_columns, original_columns) or None.
        """
        table_id = chunk.metadata.get("table_id")
        if not table_id:
            logger.warning(f"Table chunk {chunk.id} missing table_id")
            return None

        if self.table_store is None:
            logger.warning("TableStore not provided")
            return None

        # Get SQLite table name
        table_name = self.table_store.get_table_name(table_id)
        if not table_name:
            logger.warning(f"Table {table_id} not found in store")
            return None

        # Get column mappings
        columns_info = self.table_store.get_columns(table_id)
        if not columns_info:
            logger.warning(f"Could not get columns for {table_id}")
            return None

        sanitized_cols, original_cols = columns_info
        return table_name, sanitized_cols, original_cols

    def _get_sample_data(
        self, table_name: str, columns: list[str], limit: int = 3
    ) -> list[list[str]]:
        """Fetch sample data from SQLite table."""
        if self.table_store is None:
            return []

        cols_str = ", ".join(f'"{c}"' for c in columns)
        sql = f'SELECT {cols_str} FROM "{table_name}" LIMIT {limit}'

        result = self.table_store.execute_query("", sql)
        if result:
            _, rows = result
            return [[str(v) if v is not None else "" for v in row] for row in rows]
        return []

    def _process_table_chunk(self, query: str, chunk: Chunk) -> Chunk:
        """
        Process a table schema chunk using SQLite.

        1. Get table info from store
        2. LLM selects relevant columns
        3. LLM generates SQL query
        4. Execute query directly against SQLite
        5. Return augmented chunk

        Args:
            query: User query.
            chunk: Table schema chunk.

        Returns:
            Augmented chunk with query results, or original on failure.
        """
        try:
            # 1. Get table info
            table_info = self._get_table_info(chunk)
            if table_info is None:
                return chunk

            table_name, sanitized_cols, original_cols = table_info
            row_count = self.table_store.get_row_count(chunk.metadata.get("table_id", ""))

            logger.debug(
                f"Processing table {table_name}: " f"{len(sanitized_cols)} cols, {row_count} rows"
            )

            # 2. Get sample data for LLM context
            sample_rows = self._get_sample_data(table_name, sanitized_cols)

            # 3. LLM selects relevant columns (using original names for user-friendliness)
            needed_original = self._select_columns(query, original_cols, sample_rows)
            logger.debug(f"Selected columns: {needed_original}")

            # Map to sanitized column names
            col_mapping = dict(zip(original_cols, sanitized_cols))
            needed_sanitized = [col_mapping.get(c, c) for c in needed_original if c in col_mapping]

            if not needed_sanitized:
                needed_sanitized = sanitized_cols  # Fallback to all

            # 4. Generate SQL with sanitized column names
            sample_for_sql = self._get_sample_data(table_name, needed_sanitized)
            sql = self._generate_sql(query, table_name, needed_sanitized, sample_for_sql)
            logger.debug(f"Generated SQL: {sql}")

            # 5. Execute against SQLite
            result = self.table_store.execute_query(chunk.metadata.get("table_id", ""), sql)

            if result is None:
                logger.warning(f"SQL execution failed for {table_name}")
                return chunk

            col_names, rows = result

            # 6. Format results and augment chunk
            return self._augment_chunk(chunk, sql, col_names, rows, row_count)

        except Exception as e:
            logger.warning(f"Table query failed for {chunk.id}: {e}")
            return chunk

    def _needs_multi_table(self, query: str, table_chunks: list[Chunk]) -> bool:
        """Detect if query needs data from multiple tables."""
        if len(table_chunks) < 2:
            return False

        schemas = []
        for chunk in table_chunks:
            table_info = self._get_table_info(chunk)
            if table_info:
                pg_name, _, original_cols = table_info
                schemas.append(f"- {pg_name}: {', '.join(original_cols)}")

        if not schemas:
            return False

        prompt = self.MULTI_TABLE_DETECT_PROMPT.format(
            table_schemas="\n".join(schemas),
            question=query,
        )
        chat = self.chat_factory(self.TIER_COLUMN_SELECT)
        response = chat.chat([{"role": "user", "content": prompt}])
        return response.strip().lower() == "yes"

    def _process_multi_table(self, query: str, table_chunks: list[Chunk]) -> Chunk:
        """
        Process multiple tables with JOIN support.

        Generates and executes JOIN SQL directly against SQLite.

        Args:
            query: User query.
            table_chunks: Retrieved table schema chunks.

        Returns:
            Merged result chunk.
        """
        try:
            # Gather table info
            tables_info: list[tuple[str, list[str], list[str]]] = []

            for chunk in table_chunks:
                info = self._get_table_info(chunk)
                if info:
                    pg_name, sanitized, original = info
                    sample = self._get_sample_data(pg_name, sanitized)
                    tables_info.append((pg_name, sanitized, original, sample))

            if not tables_info:
                return self._process_table_chunk(query, table_chunks[0])

            # Generate multi-table SQL
            sql = self._generate_multi_table_sql(query, tables_info)
            logger.debug(f"Generated multi-table SQL: {sql}")

            # Execute against SQLite
            result = self.table_store.execute_multi_table_query(sql)

            if result is None:
                logger.warning("Multi-table SQL failed, falling back to single-table")
                return self._process_table_chunk(query, table_chunks[0])

            col_names, rows = result

            # Create merged result chunk
            table_names = [t[0] for t in tables_info]
            return self._create_multi_table_result_chunk(
                table_chunks, table_names, sql, col_names, rows
            )

        except Exception as e:
            logger.warning(f"Multi-table query failed: {e}, falling back to single-table")
            return self._process_table_chunk(query, table_chunks[0])

    def _generate_multi_table_sql(
        self,
        query: str,
        tables: list[tuple[str, list[str], list[str], list[list[str]]]],
        max_retries: int = 2,
    ) -> str:
        """Generate SQL that may JOIN multiple tables, with validation and retry."""
        previous_error = None

        for attempt in range(max_retries + 1):
            sql = self._generate_multi_table_sql_attempt(query, tables, previous_error)

            # Validate by test execution
            result = self.table_store.execute_multi_table_query(sql)
            if result is not None:
                logger.debug(
                    f"Multi-table SQL validated successfully"
                    f"{' (retry ' + str(attempt) + ')' if attempt > 0 else ''}"
                )
                return sql

            previous_error = "Query execution failed"
            logger.warning(
                f"Multi-table SQL validation failed (attempt {attempt + 1}/{max_retries + 1})"
            )

            if attempt == max_retries:
                logger.error(f"Multi-table SQL generation failed after {max_retries + 1} attempts")

        return sql

    def _generate_multi_table_sql_attempt(
        self,
        query: str,
        tables: list[tuple[str, list[str], list[str], list[list[str]]]],
        previous_error: str | None,
    ) -> str:
        """Generate multi-table SQL query (single attempt)."""
        schemas = []
        for pg_name, sanitized, original, samples in tables:
            sample_vals = {}
            for i, h in enumerate(original):
                sample_vals[h] = [row[i] for row in samples if i < len(row)]
            # Show both original and sanitized names for LLM
            col_info = [f"{orig} (column: {san})" for orig, san in zip(original, sanitized)]
            schemas.append(f"Table: {pg_name}\nColumns: {col_info}\nSamples: {sample_vals}")

        error_context = ""
        if previous_error:
            error_context = f"""
IMPORTANT: Your previous SQL query failed with this error:
{previous_error}

Please generate corrected SQL that avoids this error."""

        prompt = self.MULTI_TABLE_SQL_PROMPT.format(
            table_schemas="\n\n".join(schemas),
            question=query,
            max_results=self.max_results,
        )
        prompt = prompt + error_context

        chat = self.chat_factory(self.TIER_SQL_GENERATE)
        response = chat.chat([{"role": "user", "content": prompt}])
        return extract_sql(response)

    def _create_multi_table_result_chunk(
        self,
        source_chunks: list[Chunk],
        table_names: list[str],
        sql: str,
        col_names: list[str],
        results: list[list[Any]],
    ) -> Chunk:
        """Create merged result chunk from multi-table query."""
        results_md = self._format_as_markdown(col_names, results)

        content = f"""Multi-table query result
Source tables: {", ".join(table_names)}

SQL executed: {sql}
Results ({len(results)} rows):
{results_md}"""

        combined_id = "_".join(c.metadata.get("table_id", "unknown")[:4] for c in source_chunks)

        return Chunk(
            id=f"multi_table_{combined_id}",
            doc_id=source_chunks[0].doc_id,
            content=content,
            chunk_index=0,
            metadata={
                "is_multi_table_result": True,
                "source_tables": table_names,
                "sql_executed": sql,
                "result_count": len(results),
            },
        )

    def _select_columns(
        self, query: str, columns: list[str], sample_rows: list[list[str]] | None = None
    ) -> list[str]:
        """Use LLM to select relevant columns."""
        samples_str = "(no sample data)"
        if sample_rows:
            sample_lines = []
            for row in sample_rows[:3]:
                row_str = " | ".join(f"{col}={val}" for col, val in zip(columns, row))
                sample_lines.append(row_str)
            samples_str = "\n".join(sample_lines)

        prompt = self.COLUMN_SELECT_PROMPT.format(
            columns=columns,
            samples=samples_str,
            question=query,
        )
        chat = self.chat_factory(self.TIER_COLUMN_SELECT)
        response = chat.chat([{"role": "user", "content": prompt}])
        return self._parse_json_list(response, fallback=columns)

    def _generate_sql(
        self,
        query: str,
        table_name: str,
        columns: list[str],
        sample_rows: list[list[str]],
        max_retries: int = 2,
    ) -> str:
        """Generate SQL query with validation and retry."""
        previous_error = None

        for attempt in range(max_retries + 1):
            sql = self._generate_sql_attempt(
                query, table_name, columns, sample_rows, previous_error
            )

            # Validate by test execution
            result = self.table_store.execute_query("", sql)
            if result is not None:
                logger.debug(
                    f"SQL validated successfully"
                    f"{' (retry ' + str(attempt) + ')' if attempt > 0 else ''}"
                )
                return sql

            previous_error = "Query execution failed"
            logger.warning(f"SQL validation failed (attempt {attempt + 1}/{max_retries + 1})")

            if attempt == max_retries:
                logger.error(f"SQL generation failed after {max_retries + 1} attempts")

        return sql

    def _generate_sql_attempt(
        self,
        query: str,
        table_name: str,
        columns: list[str],
        sample_rows: list[list[str]],
        previous_error: str | None,
    ) -> str:
        """Generate SQL query (single attempt)."""
        samples: dict[str, list[str]] = {}
        for i, col in enumerate(columns):
            samples[col] = [row[i] for row in sample_rows if i < len(row)]

        error_context = ""
        if previous_error:
            error_context = f"""
IMPORTANT: Your previous SQL query failed with this error:
{previous_error}

Please generate corrected SQL that avoids this error."""

        prompt = self.SQL_PROMPT.format(
            table_name=table_name,
            columns=columns,
            samples=samples,
            question=query,
            max_results=self.max_results,
        )
        prompt = prompt + error_context

        chat = self.chat_factory(self.TIER_SQL_GENERATE)
        response = chat.chat([{"role": "user", "content": prompt}])
        return extract_sql(response)

    def _augment_chunk(
        self,
        chunk: Chunk,
        sql: str,
        cols: list[str],
        rows: list[list[Any]],
        row_count: int | None,
    ) -> Chunk:
        """Create augmented chunk with query results."""
        results_md = self._format_as_markdown(cols, rows)

        columns = chunk.metadata.get("columns", [])
        total_rows = row_count or chunk.metadata.get("row_count", "unknown")

        content = f"""Table: {chunk.doc_id}
Columns: {", ".join(columns) if columns else "see below"}
Total rows: {total_rows}

--- SQL Query Results (from FULL dataset, not just sample) ---
Query: {sql}
Results ({len(rows)} rows):
{results_md}

Note: These results are computed from all {total_rows} rows in the table. Use these results to answer the question."""

        return Chunk(
            id=chunk.id,
            doc_id=chunk.doc_id,
            content=content,
            chunk_index=chunk.chunk_index,
            metadata={
                **chunk.metadata,
                "sql_executed": sql,
                "result_count": len(rows),
            },
        )

    def _format_as_markdown(self, cols: list[str], rows: list[list[Any]]) -> str:
        """Format query results as markdown table."""
        if not rows:
            return "(no results)"

        display_rows = rows[: self.max_results]

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

        if len(rows) > self.max_results:
            lines.append(f"\n... and {len(rows) - self.max_results} more rows")

        return "\n".join(lines)

    def _parse_json_list(self, response: str, fallback: list[str]) -> list[str]:
        """Parse JSON list from LLM response."""
        parsed = parse_llm_json(response, as_array=True)
        if parsed:
            return [str(item) for item in parsed]

        logger.warning("Failed to parse column selection, using all columns")
        return fallback


__all__ = ["TableQueryStep"]
