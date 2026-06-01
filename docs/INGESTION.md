<!-- docs/INGESTION.md -->
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

### 1. File Discovery

Finds files to ingest and provides local access.

```
point(path) → recursive scan → [SourceFile, SourceFile, ...]
```

| Component | Purpose |
|-----------|---------|
| `SourceFile` | The unit of input to a parser (URI, local path, MIME type, metadata) |

**What happens:**
1. Recursively scans the input directory (`pathlib.rglob`)
2. Filters by supported extensions
3. Constructs a `SourceFile` for each discovered file

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

## Enrichment (KragIngestPipeline)

KRAG ingestion is driven by `KragIngestPipeline` — per-file operations
**parse → summarize → enrich**, then a corpus **finalize**. The
summarize and enrich steps add LLM-generated metadata.

```
┌─────────────────────────────────────────────────────────────────┐
│  KragIngestPipeline                                             │
│  per file:  parse → summarize → enrich      corpus:  finalize   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  summarize  →  1-2 sentence LLM summaries for sections / tables  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     KragEnricher                         │   │
│  │       One LLM call per batch (~15 symbols/sections)       │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │  Keywords         │  Entities          │  Temporal        │   │
│  │  (exact-match     │  (classes, people, │  (dates,         │   │
│  │  identifiers)     │  technologies)     │  versions, refs) │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  hierarchy summaries (pipeline-built when summarizer is configured) │
│    L1: one group summary per document file (on section metadata) │
│    L2: corpus summary rolled up from L1 (built during finalize)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### KragEnricher

`KragEnricher` extracts **keywords + entities + temporal metadata** in a
single LLM call per batch of ~15 symbols or sections. This makes enrichment
nearly free (~$0.13-0.74 for 1000 units). It runs when `enricher:` is
configured.

**What it extracts:**
- **Keywords** - Exact-match identifiers (TC-1001, JIRA-123, `AuthService`)
- **Entities** - Named entities (classes, people, technologies)
- **Temporal metadata** - Dates, version numbers, time references

**Results:**
- Keywords / entities stored on each symbol's / section's
  `keywords` and `entities` fields
- Temporal references stored on `metadata["temporal"]`
- Entities also fed into the `EntityGraphStore` for related-unit lookup

### Hierarchy

L1 and L2 summaries for analytical queries, built by the pipeline itself. They
run when `summarizer:` is configured, independent of `enricher:`.

**Levels:**
- **Level 1:** Per-file group summary — stored on each document
  section's `metadata["hierarchy_summary"]`
- **Level 2:** Corpus summary — rolled up from the L1 summaries during
  `finalize`, stored as a synthetic retrievable section ("Corpus
  Overview")

**Use case:** "What are the main themes?" retrieves the L2 corpus
summary instead of random sections.

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
from pathlib import Path

from fitz_sage.core import Query
from fitz_sage.engines.fitz_krag import FitzKragEngine, FitzKragConfig

cfg = FitzKragConfig(
    synthesizer="endpoint/qwen2.5-7b-instruct",
    chat_base_url="http://localhost:8080/v1",
    collection="my_collection",
)
engine = FitzKragEngine(cfg)
engine.load("my_collection")

# Register a directory for progressive querying and background indexing.
manifest = engine.point(Path("./docs"), collection="my_collection")
answer = engine.answer(Query(text="What is in these docs?"))
print(f"Registered {len(manifest.entries())} files")
print(answer.text)
```

For one-off use of a single file, pass that file path to `point()`.
Per-collection manifest and parsed caches live under the fitz workspace
collection directory.

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_sage/engines/fitz_krag/ingestion/pipeline.py` | `KragIngestPipeline` — ingestion core (parse/summarize/enrich/finalize) |
| `fitz_sage/engines/fitz_krag/ingestion/enricher.py` | `KragEnricher` — keyword/entity/temporal extraction |
| `fitz_sage/ingestion/parser/router.py` | Parser selection |
| `fitz_sage/ingestion/chunking/router.py` | Chunker selection |
| `fitz_sage/cli/commands/query.py` | CLI command (--source triggers ingestion) |

---

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) - VLM and rerank control
- [PLUGINS.md](PLUGINS.md) - Creating custom chunkers/parsers
