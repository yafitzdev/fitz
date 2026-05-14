# Unified Storage with SQLite + FTS5

> Why fitz-sage stores everything in SQLite and ranks with FTS5 `bm25()`.

---

## TL;DR

As of **v0.12.0** fitz-sage stores everything — metadata, structured tables,
keywords, full-text — in **SQLite with WAL mode and FTS5 indexes**.
One `.db` file per collection under `<workspace>/sqlite/`. Zero install,
stdlib only.

- **No vector database.** Retrieval is BM25 over FTS5 external-content
  tables, plus KRAG address routing and LLM reranking — vectors were
  removed in the same release as the embedding pipeline.
- **No server.** SQLite is a file. Open it, query it, close it.
  Each call gets its own connection (microseconds to open).
- **One file per collection.** `fitz_<collection>.db`. Delete a
  collection by `os.unlink`.
- **Stdlib + nothing.** Drops the `psycopg`, `psycopg-pool`,
  `fitz-pgserver`, `faiss-cpu`, `pgvector` chain of dependencies.

---

## How It Works

### File layout

```
~/.fitz/
└── sqlite/
    ├── fitz_default.db        # collection "default"
    ├── fitz_default.db-wal    # WAL journal
    ├── fitz_default.db-shm    # shared-memory file
    ├── fitz_codebase.db       # collection "codebase"
    └── ...
```

### Connection pragmas

Every connection opened by `SqliteConnectionManager` runs these pragmas:

```sql
PRAGMA journal_mode = WAL;        -- writers don't block readers
PRAGMA synchronous = NORMAL;      -- WAL-safe durability
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA busy_timeout = 30000;      -- 30s lock wait
```

WAL gives multi-reader / single-writer concurrency without the cost of a
server. The 30-second `busy_timeout` covers the rare contention case.

### Connection lifecycle

`SqliteConnectionManager` is a process-wide singleton, but it holds no
pool — each `with manager.connection(collection) as conn:` opens a fresh
`sqlite3.Connection` and closes it on exit. SQLite open is on the order
of tens of microseconds; pooling is unnecessary and adds lock complexity.

---

## Schema (per collection)

A collection's `.db` file holds the krag stores (sections, symbols,
import graphs, raw files, tables) plus the keyword vocabulary, all in
one file. The shapes are owned by each store; the recurring pattern is:

```sql
-- A content table (example: sections)
CREATE TABLE IF NOT EXISTS sections (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    title         TEXT,
    level         INTEGER,
    page_start    INTEGER,
    page_end      INTEGER,
    content       TEXT NOT NULL,
    summary       TEXT,
    parent_section_id TEXT,
    position      INTEGER,
    metadata      TEXT NOT NULL DEFAULT '{}'
);

-- An external-content FTS5 index over `content`
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts
USING fts5(content, content='sections', content_rowid='rowid');
```

The FTS5 table mirrors `content` from the base table; updates flow
through triggers so the index stays current without storing the text
twice on disk.

### Search

```sql
SELECT s.*, bm25(sections_fts) AS rank
FROM sections_fts
JOIN sections s ON s.rowid = sections_fts.rowid
WHERE sections_fts MATCH ?
ORDER BY rank
LIMIT ?;
```

FTS5's `bm25()` returns negative numbers (lower = better match).
The store flips the sign before returning so downstream consumers can
treat higher as better — match this convention if you add a new store.

---

## Schema port notes (PostgreSQL → SQLite)

For anyone reading old code or migrating extensions:

| PostgreSQL                                | SQLite                                      |
| ----------------------------------------- | ------------------------------------------- |
| `JSONB`                                   | `TEXT` + JSON1 (`json_extract`, `json_each`)|
| `TEXT[]`                                  | JSON arrays + `json_each`                   |
| `tsvector @@ to_tsquery(...) + ts_rank()` | FTS5 virtual table + `bm25()`               |
| `ILIKE`                                   | `LIKE COLLATE NOCASE`                       |
| `unnest(columns)`                         | `json_each(columns)`                        |
| `%s` parameter binding                    | `?` parameter binding                       |
| `DROP DATABASE`                           | `os.unlink(path)` (+ remove `-wal`/`-shm`)  |

---

## Trade-offs we accepted

| Trade-off                                  | Why it's acceptable                                                                                          |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| **No native vector search**                | fitz-sage moved to a chat-only retrieval architecture in v0.11/v0.12 — vectors were never in the hot path.   |
| **Single-writer concurrency**              | A knowledge-base workload is read-heavy; WAL handles it.                                                     |
| **No multi-host shared storage**           | Workloads that need that need a real database — out of scope for fitz-sage's single-process target.          |
| **No `DROP DATABASE` semantics**           | Collections map 1:1 to files. `delete_collection` is `os.unlink`.                                            |

---

## When this is the wrong choice

Use a different tool if any of these apply:

- You need multi-node, multi-writer access to the same store.
- Your corpus exceeds what a single SQLite file can comfortably hold
  (hundreds of GB).
- You need dedicated approximate-nearest-neighbour search at scale —
  fitz-sage's retrieval doesn't, but if yours does, route around the
  store entirely (build it on top, not inside it).

---

## Related

- [**KRAG**](krag.md) — Knowledge Routing Augmented Generation, the
  retrieval layer that sits on top of these stores.
- [**Configuration Guide**](../../CONFIG.md) — the few storage knobs
  worth knowing about.
- [**CHANGELOG**](../../../CHANGELOG.md) — the v0.12.0 entry covers the
  storage swap and what got removed alongside it.
