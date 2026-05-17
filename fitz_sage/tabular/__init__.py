# fitz_sage/tabular/__init__.py
"""
Tabular Data Routing - Store tables as structured data, query with SQL.

Three modes of operation:

1. **Embedded tables** (from documents):
   Tables extracted from markdown, PDF, etc. are stored as JSON in chunk payloads.
   Small and quick - data lives in the SQLite store.

2. **Standalone table files** (CSV, TSV):
   Tables from files are stored via TableStore.
   Schema chunk in the store points to table_id, actual data fetched at query time.

3. **Direct table query** (quickstart mode):
   Fast path for querying tables without RAG pipeline.
   Reads headers → LLM selects columns → parses only needed columns → SQL query.
   Sub-second responses for table queries.

Usage (Ingestion - embedded tables):
    from fitz_sage.tabular import TableExtractor

    extractor = TableExtractor()
    modified_doc, table_chunks = extractor.extract(parsed_doc)

Usage (Ingestion - CSV files):
    from fitz_sage.tabular.parser import parse_csv, can_parse
    from fitz_sage.tabular.store import get_table_store
    from fitz_sage.tabular.models import create_schema_chunk_for_stored_table

    if can_parse(file_path):
        parsed = parse_csv(file_path)
        store = get_table_store(collection)
        hash = store.store(parsed.table_id, parsed.columns, parsed.rows, str(file_path))
        chunk = create_schema_chunk_for_stored_table(
            table_id=parsed.table_id,
            columns=parsed.columns,
            row_count=parsed.row_count,
            source_file=str(file_path),
            table_hash=hash,
            sample_rows=parsed.rows[:3],
        )

Usage (Query via RAG):
    from fitz_sage.llm import get_chat_factory
    from fitz_sage.tabular import TableQueryStep

    factory = get_chat_factory("cohere")
    step = TableQueryStep(chat_factory=factory, table_store=store)
    augmented_chunks = step.execute(query, chunks)

Usage (Direct query - fast path):
    from fitz_sage.llm import get_chat_factory
    from fitz_sage.tabular import DirectTableQuery, is_table_file

    if is_table_file(file_path):
        factory = get_chat_factory("cohere")
        query = DirectTableQuery(chat_factory=factory)
        result = query.query(file_path, "how many cars sold in 2005?")
        print(result.answer)
"""

from fitz_sage.tabular.direct_query import (
    DirectQueryResult,
    DirectTableQuery,
    is_table_file,
)
from fitz_sage.tabular.extractor import TableExtractor
from fitz_sage.tabular.models import (
    ParsedTable,
    create_schema_chunk,
    create_schema_chunk_for_stored_table,
)
from fitz_sage.tabular.query import TableQueryStep

__all__ = [
    "ParsedTable",
    "create_schema_chunk",
    "create_schema_chunk_for_stored_table",
    "TableExtractor",
    "TableQueryStep",
    # Direct query (fast path)
    "DirectTableQuery",
    "DirectQueryResult",
    "is_table_file",
]
