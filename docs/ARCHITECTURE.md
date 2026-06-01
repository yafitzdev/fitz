<!-- docs/ARCHITECTURE.md -->
# Architecture Overview

High-level system design of fitz-sage **v0.14.1+**.

The architecture has three load-bearing decisions:

1. **Managed enrichment, optional endpoints.** Required enrichment runs through
   the in-process Qwen3.5 0.8B ONNX runtime on CPU. Fitz downloads the model on
   first ingest if missing. Optional synthesis, query intelligence, and vision
   use OpenAI-compatible HTTP endpoints or cloud/enterprise presets.
2. **No embeddings.** Retrieval is BM25 over SQLite FTS5 + KRAG
   typed-unit routing (symbols, sections, tables) + an ONNX cross-encoder
   reranker that scores candidates in a single local forward pass — no
   chat call. No vector DB, no embedding model, no `vector` column anywhere.
3. **No server.** Storage is SQLite — one `.db` file per collection
   under `<workspace>/sqlite/`. Open it, query it, close it. WAL mode
   gives multi-reader / single-writer concurrency.

---

## System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  User Interface                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                          │
│  │  CLI        │  │  Python SDK │  │  REST API   │                          │
│  │  fitz ...   │  │  import ... │  │  /query     │                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Runtime (engine orchestrator: load config → build engine → dispatch)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Engine: FitzKRAG                                                           │
│  - Deterministic planner + optional query-intelligence provider             │
│  - Router: symbol search · section search · table SQL                       │
│  - Expander (import graph, entity links, same-file refs, hierarchy)         │
│  - ONNX cross-encoder reranker (gte-reranker-modernbert-base)               │
│  - Optional synthesizer (chat call that writes the answer)                  │
│  - Governance (pyrrho → TRUSTWORTHY / DISPUTED / ABSTAIN)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│  LLM / ONNX         │  │  Storage (SQLite)   │  │  Ingestion Pipeline     │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────────┤
│  Qwen3.5 0.8B ONNX  │  │  WAL + FTS5         │  │  Parse (Docling / OCR)  │
│  for enrichment     │  │  one .db per        │  │  Chunk (semantic +      │
│  endpoint/cloud     │  │  collection         │  │   structured)           │
│  for optional chat  │  │  bm25() ranking     │  │  Required enrich        │
│  + enterprise auth  │  │  json_each, json1   │  │   keywords, entities)   │
└─────────────────────┘  └─────────────────────┘  └─────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  External optional endpoints: vLLM · Ollama · LM Studio · OpenAI · ...     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Dependencies

Strict import rules enforce separation of concerns (verified by
`python -m tools.contract_map --fail-on-errors`):

| Layer            | May import from                              |
| ---------------- | -------------------------------------------- |
| `core/`          | nothing (no imports from engines/ingestion)  |
| `retrieval/`     | `core/`                                      |
| `llm/`           | `core/`                                      |
| `storage/`       | `core/`                                      |
| `ingestion/`     | `core/`                                      |
| `engines/`       | `core/`, `llm/`, `storage/`, `retrieval/`    |
| `runtime/`       | all layers                                   |
| `cli/`, `api/`   | all layers                                   |

---

## Data Flow

### Query

Retrieval runs as a tiered pipeline. Tiers 2-5 form one `RetrievalPass`
(retrieve → rerank → read); multi-hop loops the pass on a bridge query
when pyrrho judges the evidence insufficient.

```
Tier 1  Transform   deterministic plan → optional rewrite/analyze/detect intent
Tier 2  Generate    route to symbol / section / table search over FTS5 bm25
Tier 3  Fuse        merge across strategies, dedup, keyword-boost
Tier 4  Rerank      ONNX cross-encoder (gte-reranker-modernbert-base, ~30 ms CPU)
Tier 5  Read        fetch content for the surviving addresses
        expand      import graph, entity links, hierarchical context
Tier 6  Govern      pyrrho → AnswerMode ∈ {TRUSTWORTHY, DISPUTED, ABSTAIN}
        synthesize  optional chat call writes the answer + provenance
```

### Ingestion

```
Files → Parse (Docling for PDF/DOCX, GLM-OCR for scans, tree-sitter
        for code, native parsers for CSV/XLSX/SQL/JSON)
      → Chunk (sections, symbols, table rows — typed units, not
        fixed-size windows)
      → Required enrich (LLM-generated summaries, keywords, named entities,
        hierarchical L1/L2 summaries)
      → Index into per-collection SQLite + FTS5 external-content tables
```

---

## Chat Provider Model

The LLM layer has one managed local enrichment runtime — Qwen3.5 0.8B ONNX —
and one canonical optional endpoint provider — **`endpoint`**:

| Spec                                   | Resolves to                                              |
| -------------------------------------- | -------------------------------------------------------- |
| `onnx/qwen3.5-0.8b`                    | managed Qwen3.5 0.8B ONNX generation on CPU             |
| `endpoint/<URL>/<model>` or YAML triple | `chat_base_url` + `model` + optional `chat_api_key_env` |
| `openai/<model>`                       | endpoint pointing at `https://api.openai.com/v1`         |
| `azure_openai/<deployment>`            | endpoint with Azure deployment URL                       |
| `enterprise/<provider>/<model>`        | endpoint + M2M / mTLS / custom-CA auth                   |

Legacy names `ollama`, `cohere`, `anthropic` were removed in v0.12.0
and raise `ValueError` with migration text.

The **`OnnxReranker`** is a separate INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default) — same
architecture family as pyrrho, and both run on raw `onnxruntime`
via the shared `OnnxEncoderBackend`. It scores `(query, candidate)`
pairs locally in ~30–100 ms on CPU with no external LLM call.
See [features/retrieval/reranking.md](features/retrieval/reranking.md).

---

## Storage Model

One **SQLite** file per collection:

```
<workspace>/sqlite/
├── fitz_default.db
├── fitz_default.db-wal
├── fitz_default.db-shm
└── fitz_<other>.db
```

Connections are opened per call with these pragmas:

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 30000;
```

Each store inside the `.db` (sections, symbols, table-store, import
graph, vocabulary, entity graph) has an FTS5 external-content companion
indexed over its searchable columns. `search_bm25()` queries the FTS5
table and joins back; the raw `bm25()` value (negative = better) is
sign-flipped so downstream consumers treat higher as better.

See [features/platform/unified-storage.md](features/platform/unified-storage.md)
for the full schema-port notes (PostgreSQL → SQLite).

---

## Feature Control

Features are controlled by **provider presence**, not boolean flags. Enrichment
is standard engine behavior and is required for ingestion:

```yaml
# ENABLED — a provider is named
rerank: onnx
governance: pyrrho
synthesizer: endpoint/qwen2.5-7b-instruct
chat_base_url: http://localhost:8080/v1

# DISABLED — omit the key (or set null)
# synthesizer: null → no answer generation
# rerank: null → no reranking step
# governance: null → no governance
```

---

## Core Types

```python
@dataclass
class Query:
    text: str
    metadata: dict | None = None


@dataclass
class Answer:
    text: str
    mode: AnswerMode              # TRUSTWORTHY | DISPUTED | ABSTAIN
    provenance: list[Provenance]  # source attribution chain
    metadata: dict
```

There is no public `Chunk` type in the retrieval path — KRAG uses
**typed units** (`Symbol`, `Section`, `TableSpec`) with structural
metadata, not fixed-size text windows.

---

## Configuration Layout

```
~/.fitz/
├── config/
│   └── fitz_krag.yaml       # engine config (role providers, retrieval knobs)
├── sqlite/                  # one .db per collection
│   ├── fitz_default.db
│   └── ...
└── ingest_state.json        # incremental ingest manifest
```

Minimal local config:

```yaml
collection: default
parser: cpu
rerank: onnx
governance: pyrrho
query_intelligence: null
synthesizer: null
chat_base_url: http://127.0.0.1:8080/v1
```

Override per-invocation:

```bash
fitz answer "..." \
  --endpoint https://api.together.xyz/v1 \
  --synthesizer endpoint/meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY
```

---

## Directory Structure

```
fitz_sage/
├── core/                # Foundation layer (Query, Answer, Provenance, protocols)
├── engines/
│   └── fitz_krag/       # KRAG engine: retrieval/, generation/, ingestion/
├── retrieval/           # SHARED retrieval intelligence
│   ├── detection/       # LLM-based query classification
│   ├── entity_graph/    # Entity-based linking
│   ├── vocabulary/      # Keyword storage + matching
│   └── rewriter/        # LLM-based query rewriting
├── llm/                 # Managed ONNX enrichment + optional chat endpoints
│   ├── providers/       # onnx_chat, endpoint, enterprise, onnx_reranker
│   ├── auth/            # ApiKeyAuth, M2MAuth, CompositeAuth
│   ├── config.py        # provider-spec → instance factory
│   └── client.py        # get_chat, ...
├── storage/             # SqliteConnectionManager (WAL, FTS5)
├── ingestion/           # parser plugins, chunking plugins, enrichment
├── tabular/             # CSV/XLSX → SqliteTableStore + SQL generation
├── governance/          # pyrrho classifier, answer modes, instructions
├── runtime/             # multi-engine orchestration
├── cli/                 # typer commands
├── api/                 # FastAPI app + routes
└── sdk/                 # stateful Python interface
```

---

## Design Principles

1. **Explicit over clever.** No magic. Read the config; know what happens.
2. **Managed enrichment.** Required metadata generation uses local Qwen ONNX;
   optional chat calls go through `endpoint` or its presets.
3. **Structure-first retrieval.** Parse code/docs into typed units at
   ingest; route to the right strategy at query time.
4. **No embeddings.** BM25 + KRAG routing + ONNX rerank covers the
   ground the embedding stack used to. Confirmed against fitz-gov v5.
5. **Honest over helpful.** Say `ABSTAIN` instead of hallucinating.
6. **Files over frameworks.** Plugins are Python modules wired by config,
   not framework abstractions.
7. **Local-first.** SQLite + ONNX enrichment works offline after the model is
   cached; endpoint servers are optional.
8. **Provenance always.** Every answer traces back to source addresses.

---

## See Also

- [Unified Storage](features/platform/unified-storage.md) — SQLite + FTS5
- [PLUGINS.md](PLUGINS.md) — plugin development guide
- [CONFIG.md](CONFIG.md) — configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) — feature-control architecture
- [INGESTION.md](INGESTION.md) — ingestion pipeline
- [CONSTRAINTS.md](CONSTRAINTS.md) — epistemic guardrails
- [features/platform/openai-compatible-endpoint.md](features/platform/openai-compatible-endpoint.md) — the canonical chat provider
