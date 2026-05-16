# Progressive KRAG: Zero-Wait Querying with Agentic Search

## Overview

Progressive KRAG eliminates the ingestion barrier entirely. Instead of requiring users to run a separate ingest command (which takes minutes for LLM summarisation) before they can ask questions, users point at a folder and query immediately. An LLM-driven agentic search handles unindexed files on the fly, while a background worker silently indexes everything. Queries get progressively faster as indexing completes — but they work from second one.

**Before**: `fitz ingest ./docs` (wait minutes) -> `fitz query "question"`
**After**: `fitz query --source ./docs "question"` (works immediately)

## User-Facing Changes

### Pointing at a source

The user-facing entry point is `fitz query --source ./docs "question"`:

```bash
fitz query --source ./my-codebase "how does authentication work?"
# Registers the directory, then answers immediately --
# even before any file is indexed. Queries get faster over time.
```

There is no separate `fitz point` command. `point()` exists as an internal engine/service method, not a CLI command. The old `fitz ingest` command, its CLI subcommands, and its SDK/service methods have been removed.

### SDK

```python
from fitz_sage.sdk import Fitz

fitz = Fitz()
answer = fitz.query("how does X work?", source="./docs")  # works immediately
```

### API

The REST API's `/query` endpoint accepts an optional `source` field to register documents before querying.

## Architecture

```
fitz query --source ./docs "how does X work?"
  <- builds manifest, starts background worker, answers immediately

Query Router
+-- Indexed path (files at SUMMARIZED state) -> BM25 / FTS5 strategies
+-- Agentic path (files not yet indexed)     -> LLM picks from manifest, reads from disk
+-- Merge results -> standard KRAG generation pipeline
```

As the background worker progresses files through the state machine, they migrate from the agentic path to the indexed path automatically. The system converges to full indexed performance without user intervention.

## Components

### 1. FileManifest (`progressive/manifest.py`)

Thread-safe manifest with JSON persistence at `~/.fitz/collections/{collection}/manifest.json`. Tracks every file in the pointed directory with its indexing state, extracted symbols/headings, and priority.

**Data model**:

| Field | Description |
|-------|-------------|
| `file_id` | UUID, stable across re-scans |
| `rel_path` | Relative path from source root |
| `content_hash` | SHA-256 for change detection |
| `file_type` | Extension (`.py`, `.md`, etc.) |
| `state` | `REGISTERED` -> `PARSED` -> `SUMMARIZED` (terminal) |
| `symbols` | AST-extracted symbols (code files) |
| `headings` | Regex-extracted headings (doc files) |
| `priority` | 1-4, boosted when user queries related files |

**Thread safety**: All mutations guarded by `threading.Lock`. The query thread and background worker thread access the manifest concurrently.

### 2. ManifestBuilder (`progressive/builder.py`)

Fast directory scanner that builds the manifest without any LLM calls or database access. Runs in <500ms for 100 files.

**Extraction**: Reuses existing ingestion strategies for symbol extraction:
- `PythonCodeIngestStrategy` for `.py` (stdlib `ast`, ~50ms/file)
- `TypeScriptIngestStrategy` for `.ts`/`.tsx`/`.js`/`.jsx` (tree-sitter)
- `JavaIngestStrategy` for `.java` (tree-sitter)
- `GoIngestStrategy` for `.go` (tree-sitter)
- Regex heading extraction for `.md`/`.rst`/`.txt`

Non-Python strategies use lazy `try/except` imports -- if tree-sitter grammars aren't installed, extraction returns an empty list and the file is still registered (just without symbols).

### 3. AgenticSearchStrategy (`retrieval/strategies/agentic_search.py`)

LLM-driven file selection for files not yet at SUMMARIZED state. This is what makes queries work before indexing completes.

**Retrieval flow**:
1. Get files NOT at SUMMARIZED state from manifest
2. Build compact manifest text (~50-100 tokens/file with symbols, headings, paths)
3. If >50 unindexed files: BM25 pre-filter to top 50
4. Single LLM call (via `chat_factory("fast")`) picks 5-10 candidate files
5. Read file content from disk
6. Create `Address` objects:
   - Code files with symbols -> one `SYMBOL` address per symbol with AST line ranges
   - Doc files with headings -> one `FILE` address with heading metadata
   - Other files -> one `FILE` address

**BM25 pre-filter**: Pure Python, no dependencies. Tokenizes query and manifest text, scores by term frequency with TF saturation. Keeps the LLM prompt under token budget for large repos.

**Integration**: Agentic addresses are standard `Address` objects with `metadata.agentic = True`. They flow through the existing dedup, ranking, and filtering pipeline without modification. The `ContentReader` has a disk fallback (`source_dir` parameter) to read file content for agentic addresses that aren't yet in the database.

### 4. BackgroundIngestWorker (`progressive/worker.py`)

Daemon thread that indexes files through a two-phase state machine. Each phase processes files in priority order (queried files first).

| Phase | Transition | What happens | LLM? |
|-------|------------|--------------|------|
| 1 | REGISTERED -> PARSED | Store raw content, extract symbols/imports via strategies, write to SQLite | No |
| 2 | PARSED -> SUMMARIZED | Generate LLM summaries in batches. **Pauses during active queries.** | Yes |

**Priority queue**:
- P1: Files the user just queried about
- P2: Files in the same directory as queried files
- P3: Small files (<10KB, quick wins)
- P4: Remaining files by size ascending

**Query coordination**: When a query arrives, the engine calls `signal_query_start()` which pauses LLM summarization (phase 2) so the query gets full LLM priority. After the query completes, `signal_query_end()` resumes the worker.

**Multi-language support**: Phase 1 uses lazy-initialized strategies for Python, TypeScript/JavaScript, Java, and Go. Each strategy extracts symbols and imports which are stored through the same unified path (symbol store + import store).

### 5. Hybrid Retrieval Router Integration

The `RetrievalRouter` accepts an `agentic_strategy` parameter. During retrieval, it runs the agentic strategy alongside existing indexed strategies (code, section, table). Results from all strategies are merged, deduped, and ranked uniformly.

```
Query -> RetrievalRouter
         +-- CodeSearchStrategy    (indexed symbols)
         +-- SectionSearchStrategy (indexed sections)
         +-- TableSearchStrategy   (indexed tables)
         +-- AgenticSearchStrategy (unindexed files)
         -> merge + dedup + rank
```

As files get indexed by the background worker, they stop appearing in the agentic path and start appearing in the indexed path -- the transition is seamless.

### 6. ContentReader Disk Fallback

The `ContentReader` now accepts a `source_dir: Path` parameter. When an address references a file not yet in the database (`RawFileStore.get()` returns None) and `source_dir` is set, the reader falls back to reading directly from disk using the `disk_path` from the address metadata. This enables the full read -> expand -> generate pipeline to work for agentic results.

## Files

### New files (5)

| File | Purpose |
|------|---------|
| `engines/fitz_krag/progressive/__init__.py` | Package exports |
| `engines/fitz_krag/progressive/manifest.py` | FileManifest, ManifestEntry, FileState, ManifestSymbol, ManifestHeading |
| `engines/fitz_krag/progressive/builder.py` | ManifestBuilder -- fast AST + heading extraction, no LLM |
| `engines/fitz_krag/progressive/worker.py` | BackgroundIngestWorker -- daemon thread, 2-phase state machine |
| `engines/fitz_krag/retrieval/strategies/agentic_search.py` | AgenticSearchStrategy -- LLM-driven file selection |

### Modified files (6)

| File | Change |
|------|--------|
| `engines/fitz_krag/engine.py` | Remove `ingest()`, add `point()`, wire agentic strategy + worker, add query signaling |
| `engines/fitz_krag/retrieval/router.py` | Accept `agentic_strategy` param, run alongside indexed strategies |
| `engines/fitz_krag/retrieval/reader.py` | Accept `source_dir` param, disk fallback for unindexed files |
| `services/fitz_service.py` | Remove `ingest()`/`IngestResult`, add `point()` |
| `sdk/fitz.py` | `query()` gains a `source=` param (registers the path before querying) |
| `cli/cli.py` | `query` command gains `--source`/`-s` option |

### Deleted files (5)

| File | Reason |
|------|--------|
| `cli/commands/ingest.py` | No longer user-facing |
| `cli/commands/ingest_runner.py` | No longer user-facing |
| `cli/commands/ingest_helpers.py` | No longer user-facing |
| `cli/commands/ingest_engines.py` | No longer user-facing |
| `cli/commands/ingest_direct.py` | No longer user-facing |

### Retained internally (not modified)

The ingestion pipeline code (`ingestion/pipeline.py`, `ingestion/strategies/`, `ingestion/*_store.py`) is retained and used by the background worker. It is no longer user-facing.

## Design Decisions

**Why remove `ingest` instead of keeping both?** The `point` workflow strictly dominates: it does everything `ingest` did (via the background worker) plus it lets queries work immediately. Keeping `ingest` would mean maintaining two paths that converge to the same outcome, with the slower one offering no advantage.

**Why a BM25 pre-filter?** A pure-Python token-overlap scorer with
no dependencies is fast enough for 500+ files and keeps the LLM prompt
under budget. Since v0.12.0 dropped embeddings entirely, BM25 is also
the index used throughout the rest of retrieval — keeping the
agentic-search pre-filter aligned with the steady-state path.

**Why daemon thread instead of async?** The background worker needs to coordinate with the query thread (pause during LLM calls, boost priority). `threading.Event` provides simple, correct coordination without introducing async complexity into the existing synchronous engine.

**Why neutral scores (0.5) for agentic addresses?** The LLM already selected these files as relevant. The exact ranking within agentic results matters less than ensuring they participate in the merged result set. The existing dedup and ranking pipeline handles the rest.

**Graceful degradation for tree-sitter**: Non-Python strategies use lazy imports. If `tree-sitter-typescript`, `tree-sitter-java`, or `tree-sitter-go` aren't installed, symbol extraction silently returns empty lists. Files are still registered and searchable by path -- they just lack fine-grained symbol addresses until the background worker indexes them.
