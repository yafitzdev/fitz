# Native Table Routing

Fitz-Sage gives configured delimited files a native table path instead of
treating every row as prose. The default native extensions are `.csv` and
`.tsv`.

Embedded tables in PDF, DOCX, PPTX, Markdown, or HTML remain part of document
sections. XLSX is not a native default table format.

## Indexing

For each native table, source indexing stores:

1. raw source identity and bytes;
2. table name, original/sanitized columns, and row count;
3. a concrete SQLite table containing all parsed rows;
4. row text in `_table_row_fts` for BM25 value lookup.

The first parsed row is the schema header. Fitz-Sage does not guess past blank
or title rows, and SQLite's column limit still applies to ultra-wide exports.
Those inputs must be cleaned or reshaped by the user.

## Recall

`TableSearchStrategy` merges three bounded signals:

| Signal | Purpose |
|---|---|
| Table name and columns | Find a likely table from its schema |
| Row-value BM25 | Surface a table because a concrete value matches |
| Deterministic row scan | Handle explicit record/property, filter, or superlative shapes |

The row scan is not run for every prose query. It is reserved for table-required
profiles, exact identifiers, or clear row/property wording and is bounded by
table and row limits.

## Reading Table Evidence

Without any chat provider, the standard path can:

- look up exact identifiers across all rows;
- reuse row numbers found by row-value BM25;
- apply deterministic equality/status predicates;
- execute supported sort/superlative plans against the full table;
- fall back to a bounded scan when no stronger lookup is available.

The returned evidence states whether rows came from exact lookup, BM25 lookup,
full deterministic execution, or a bounded scan.

## Optional Generated SQL

If `chat_fast`, `chat_balanced`, or `chat_smart` is configured, the engine can
ask the balanced tier to generate a read-only query for one retrieved table.
Generated SQL is validated and executed against that table; failures fall back
to deterministic row grounding.

Setting only `synthesizer:` in the YAML file does not enable table SQL. The
standalone `fitz answer` flags populate the chat tiers for that invocation.

Fitz-Sage does not currently orchestrate generated joins across multiple native
tables.

## Configuration

```yaml
collection: reports
table_extensions: [".csv", ".tsv"]
max_table_results: 100
```

`table_extensions` changes which delimited inputs are routed through the native
parser. It does not add an XLSX parser or turn embedded document tables into
native SQLite tables.

## Boundaries

- Header repair and schema inference beyond the first parsed row are user-owned.
- Separator variants in record identifiers are not normalized.
- Bounded scans can miss a row when no literal value, schema match, or supported
  deterministic predicate points to it.
- Complex calculations and multi-table reasoning require an application-owned
  query layer or a configured optional model path.

## Implementation

- `fitz_sage/tabular/parser/`
- `fitz_sage/tabular/store/sqlite.py`
- `fitz_sage/engines/fitz_krag/ingestion/table_store.py`
- `fitz_sage/engines/fitz_krag/retrieval/strategies/table_search.py`
- `fitz_sage/engines/fitz_krag/retrieval/table_handler.py`

## Related

- [Sparse Search](../retrieval/sparse-search.md)
- [Limitations](../../LIMITATIONS.md)
