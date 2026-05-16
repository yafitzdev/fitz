# Ingestion Pipeline

How documents flow through Fitz from files to searchable chunks.

---

## Overview

The ingestion pipeline transforms your documents into searchable knowledge:

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────────────────┐
│  Files  │ →  │  Parse  │ →  │  Chunk  │ →  │ Index (SQLite + FTS5)│
└─────────┘    └─────────┘    └─────────┘    └──────────────────────┘
                   │              │
              ParsedDoc       Chunks[]
```

**Key features:**
- **Incremental** - Only processes new/changed files
- **Format-aware** - PDFs, code, markdown handled differently
- **Enrichable** - Optional LLM enhancement (summaries, hierarchies)

---

## Pipeline Stages

### 1. Source (File Discovery)

Finds files to ingest and provides local access.

```
Source.discover(path) → [SourceFile, SourceFile, ...]
```

| Component | Purpose |
|-----------|---------|
| `FileSystemSource` | Local filesystem discovery |
| `SourceFile` | Abstraction for file access (URI, local path, metadata) |

**What happens:**
1. Recursively scans the input path
2. Filters by supported extensions
3. Returns `SourceFile` objects with metadata

---

### 2. Diff (Change Detection)

Determines which files need processing.

```
Differ.diff(files, state) → (to_ingest, to_skip, to_delete)
```

| Component | Purpose |
|-----------|---------|
| `FileScanner` | Computes file hashes |
| `Differ` | Compares against stored state |
| `IngestStateManager` | Persists file hashes and chunker IDs |

**Change detection:**

| Change Type | Detection Method | Action |
|-------------|------------------|--------|
| New file | Not in state | Ingest |
| Modified file | Content hash changed | Re-ingest |
| Config changed | Chunker ID changed | Re-chunk |
| Deleted file | In state, not on disk | Mark deleted |
| Unchanged | Hash + chunker match | Skip |

**State file:** `.fitz/ingest_state.json`

```json
{
  "files": {
    "/path/to/doc.md": {
      "content_hash": "abc123...",
      "chunker_id": "recursive:1000:200",
      "chunk_ids": ["chunk1", "chunk2"],
      "ingested_at": "2024-01-15T10:30:00"
    }
  }
}
```

---

### 3. Parse (Document Extraction)

Converts files to structured documents.

```
Parser.parse(SourceFile) → ParsedDocument
```

| Component | Purpose |
|-----------|---------|
| `ParserRouter` | Routes files to parsers by extension |
| `DoclingParser` | PDFs, DOCX, images via Docling |
| `DoclingVisionParser` | Same + VLM for figure descriptions |
| `GlmOcrParser` | Hybrid pypdfium2 + GLM-OCR via the vision endpoint |
| `PlainTextParser` | Text files, markdown, code |

**Parsed document structure:**

```python
ParsedDocument(
    source_uri="file:///path/to/doc.pdf",
    elements=[
        TextElement(text="Chapter 1", level=1),
        TextElement(text="Introduction paragraph..."),
        TableElement(data=[...]),
        ImageElement(description="[Figure]"),  # or VLM description
    ],
    metadata={"title": "Document Title", ...}
)
```

**Parser selection:**

| Extension | Parser | Notes |
|-----------|--------|-------|
| `.pdf`, `.docx`, `.pptx` | Docling | Structure extraction |
| `.png`, `.jpg` | Docling | OCR + optional VLM |
| `.md`, `.txt`, `.py` | PlainText | Direct text reading |

**VLM for figures:** Set `parser: docling_vision` to enable AI-generated figure descriptions instead of `[Figure]` placeholders.

---

### 4. Chunk (Text Splitting)

Splits documents into retrieval-sized pieces.

```
Chunker.chunk(ParsedDocument) → [Chunk, Chunk, ...]
```

| Component | Purpose |
|-----------|---------|
| `ChunkingRouter` | Routes by extension or uses default |
| `RecursiveChunker` | General-purpose, respects structure |
| `MarkdownChunker` | Header-aware splitting |
| `PythonCodeChunker` | AST-aware, keeps functions intact |

**Chunk structure:**

```python
Chunk(
    id="abc123...",           # Deterministic hash
    content="The text...",    # Chunk content
    metadata={
        "source_file": "/path/to/doc.pdf",
        "chunk_index": 0,
        "page": 1,
        "heading": "Chapter 1",
    }
)
```

**Chunker ID:**

Each chunker has a unique ID encoding its parameters:

```
recursive:1000:200  # plugin:chunk_size:overlap
python_code:1500:100
markdown:800:150
```

If the chunker ID changes (e.g., you change chunk_size), affected files are automatically re-chunked.

---

### 5. Store (SQLite + FTS5)

Persists ingested units for retrieval. There are no vectors and no
database server — each collection is a single `.db` file.

| Component | Purpose |
|-----------|---------|
| `SqliteConnectionManager` | One `.db` file per collection under `<workspace>/sqlite/` |
| `SymbolStore` | Code symbols → `krag_symbol_index` (+ `krag_symbol_fts`) |
| `SectionStore` | Document sections → `krag_section_index` (+ `krag_section_fts`) |
| `TableStore` | Table metadata → `krag_table_index` |

**What happens:**
- Ingested units (code symbols, document sections, tables, chunks) are
  written to per-collection SQLite databases.
- FTS5 indexes back BM25 keyword search over symbol and section text.
- One `fitz_<collection>.db` lives under the workspace `sqlite/`
  directory — no server, no connection string, no vectors.

---

## Enrichment Pipeline (Always On)

The enrichment pipeline adds LLM-generated enhancements. **All chunk-level enrichment is baked in** - no configuration needed.

```
┌─────────────────────────────────────────────────────────────────┐
│  Enrichment Pipeline (always on, runs after chunking)           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │            ChunkEnricher (Enrichment Bus)                │   │
│  │         One LLM call per batch (~15 chunks)              │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Summary    │  Keywords     │  Entities   │ ContentType  │   │
│  │  Module     │  Module       │  Module     │ Module       │   │
│  │  (per-chunk │  (exact-match │  (classes,  │ (narrative/  │   │
│  │  summaries) │  identifiers) │  people)    │ structured)  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│  ┌───────────────────────────┴───────────────────────────┐      │
│  │                Hierarchy Enricher                      │      │
│  │  Level 0: Chunks (with enrichments from above)         │      │
│  │  Level 1: Group summaries (per source file)            │      │
│  │  Level 2: Corpus summary (all documents)               │      │
│  └────────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### ChunkEnricher (Baked In)

The `ChunkEnricher` bus extracts **summary + keywords + entities + content type** in a single LLM call per batch of ~15 chunks. This makes enrichment nearly free (~$0.13-0.74 for 1000 chunks).

**What it extracts:**
- **Summaries** - Natural language descriptions for each chunk
- **Keywords** - Exact-match identifiers (TC-1001, JIRA-123, `AuthService`)
- **Entities** - Named entities (classes, people, technologies)
- **Content Type** - Classification (`narrative`/`structured`/`technical`/`mixed`)

**Results:**
- Summaries stored in `chunk.metadata["summary"]`
- Keywords saved to `VocabularyStore` for exact-match retrieval
- Entities stored in `chunk.metadata["entities"]`
- Content type stored in `chunk.metadata["content_type"]`

### Hierarchy

Multi-level summaries for analytical queries. **Always on by default.**

```yaml
enrichment:
  hierarchy:
    group_by: source_file
```

**Levels:**
- **Level 0:** Original chunks (with summary, keywords, entities)
- **Level 1:** Group summaries (e.g., per-file)
- **Level 2:** Corpus summary (all groups)

**Use case:** "What are the main themes?" retrieves L1/L2 summaries instead of random chunks.

---

## Incremental Ingestion

Fitz only processes what's changed. When you point Fitz at a source directory, ingestion happens automatically in the background:

```bash
$ fitz query --source ./docs "What is quantum computing?"

Pointing at ./docs...
  Scanning... 847 files
  → 12 new files
  → 3 modified files
  → 832 unchanged (skipped)
  Ingesting 15 files in background...

Answer: Quantum computing uses qubits...
```

### What triggers re-ingestion?

| Change | Re-ingest? | Why |
|--------|------------|-----|
| File content changed | Yes | Content hash differs |
| Chunk size changed | Yes | Chunker ID differs |
| New file added | Yes | Not in state |
| File deleted | Mark deleted | Clean up indexed entries |

---

## File Format Support

### Documents

| Format | Parser | Features |
|--------|--------|----------|
| PDF | Docling | Tables, figures, sections |
| DOCX | Docling | Styles, tables |
| PPTX | Docling | Slides as sections |
| HTML | Docling | Structure extraction |

### Code

| Format | Chunker | Features |
|--------|---------|----------|
| Python | `python_code` | AST-aware, preserves functions |
| Markdown | `markdown` | Header-aware splitting |
| Other code | `recursive` | Respects indentation |

### Text

| Format | Parser | Notes |
|--------|--------|-------|
| `.txt` | PlainText | Direct reading |
| `.md` | PlainText | Preserves formatting |
| `.json`, `.yaml` | PlainText | Treated as text |

---

## CLI Usage

Ingestion is triggered automatically when you point Fitz at a source directory. There is no separate `ingest` command.

```bash
# Point at docs and query (ingestion happens in background)
fitz query --source ./docs "What is quantum computing?"

# Subsequent queries reuse the already-ingested data
fitz query "Explain entanglement"
```

---

## Python API

```python
import fitz_sage

# Query with source (ingestion happens in background)
answer = fitz_sage.query("What is quantum computing?", source="./docs")
```

### Advanced usage

Drive ingestion directly through the engine for fine-grained control:

```python
from fitz_sage.engines.fitz_krag import FitzKragEngine, FitzKragConfig

cfg = FitzKragConfig(
    chat_fast="endpoint",
    chat_balanced="endpoint",
    chat_smart="endpoint",
    chat_base_url="http://localhost:8080/v1",
    chat_smart_model="qwen2.5-7b-instruct",
    collection="my_collection",
)
engine = FitzKragEngine(cfg)

# Ingest a directory; incremental by default (skips unchanged files).
summary = engine.ingest("./docs")
print(f"Ingested {summary.documents} documents, {summary.symbols} symbols")
```

For one-off ingest of a single file, pass the path; for force-reingest,
pass `force=True`. Incremental state is tracked in
`.fitz/ingest_state.json` (per-file mtime + content hash).

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_sage/ingestion/diff/executor.py` | Main orchestrator |
| `fitz_sage/ingestion/parser/router.py` | Parser selection |
| `fitz_sage/ingestion/chunking/router.py` | Chunker selection |
| `fitz_sage/ingestion/state/manager.py` | State persistence |
| `fitz_sage/ingestion/enrichment/pipeline.py` | Enrichment orchestrator |
| `fitz_sage/cli/commands/query.py` | CLI command (--source triggers ingestion) |

---

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) - VLM and rerank control
- [PLUGINS.md](PLUGINS.md) - Creating custom chunkers/parsers
