# Unified SQLite Storage

Fitz-Sage stores each collection in one SQLite database with WAL mode. It does
not require a database server, search service, vector extension, or embedding
column.

## Workspace Layout

```text
<current-directory>/.fitz/
├── config.yaml
├── sqlite/
│   ├── fitz_default.db
│   ├── fitz_default.db-wal
│   └── fitz_default.db-shm
└── collections/
    └── default/
        └── manifest.json
```

Use the SDK, service, CLI, or REST collection-delete operation rather than
manually deleting only the main database file; SQLite sidecars and collection
manifest state belong to the same collection lifecycle.

## Connection Contract

`SqliteConnectionManager` opens a connection per operation and applies:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA busy_timeout = 30000;
```

WAL supports concurrent readers and a single writer. This remains a local
single-database concurrency model, not multi-host shared storage.

## Stored Surfaces

| Surface | Main storage | Search path |
|---|---|---|
| Raw source | `krag_raw_files` | addressed by typed units |
| Document sections | `krag_section_index` | external-content `krag_section_fts` over title/content |
| Code symbols | `krag_symbol_index` | external-content `krag_symbol_fts` over derived `index_text` plus name lookup |
| Import graph | `krag_import_graph` | ordinary indexes |
| Table metadata | `krag_table_index`, `_table_metadata` | name/column lookup |
| Native table rows | one generated SQLite table per source table | `_table_row_fts` plus deterministic SQL/select helpers |
| Entity graph | `entities`, `entity_chunks` | ordinary indexes |

There is no keyword-vocabulary table. Literal source terms are indexed in their
own section, symbol, or row surfaces; semantic query terms are generated at
query time.

## BM25 Convention

FTS5's `bm25()` returns lower, usually negative values for better matches.
Fitz-Sage store methods negate that value before returning candidates so the
rest of the retrieval pipeline can treat larger scores as better.

Free-form queries are not passed through as an FTS query language. The store
helper extracts word tokens, quotes them, and OR-joins them before `MATCH`.

## Tradeoffs

| Boundary | Consequence |
|---|---|
| One SQLite writer | Ingestion is serialized per collection |
| Local files | No transparent multi-node sharing |
| Sparse indexes | Different vocabulary needs Qwen query terms, corpus evidence, or user preprocessing |
| Concrete native rows | Very wide or malformed CSV/TSV inputs can exceed parser/SQLite limits |
| No vector index | Semantic recall remains a lexical bridge rather than dense nearest-neighbor search |

The measured storage and throughput behavior is recorded in
[Benchmarks](../../BENCHMARK.md); it is not a database-size or latency SLA.

## Related

- [KRAG](krag.md)
- [Sparse Search](../retrieval/sparse-search.md)
- [Native Table Routing](../ingestion/tabular-data-routing.md)
- [Configuration](../../CONFIG.md)
