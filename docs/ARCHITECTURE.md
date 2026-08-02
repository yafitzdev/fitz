<!-- docs/ARCHITECTURE.md -->
# Architecture Overview

High-level system design of fitz-sage.

The architecture has three load-bearing decisions:

1. **Immediate source index, deferred enrichment.** `point()` parses and stores
   searchable source without loading Qwen. Managed Qwen supplies standard
   query-time semantic keywords and optional background entity, hierarchy, and
   demand-summary work. Optional synthesis, query intelligence, and vision use
   OpenAI-compatible HTTP endpoints or cloud/enterprise presets.
2. **No embeddings.** Retrieval is BM25 over SQLite FTS5 + KRAG
   typed-unit routing (symbols, sections, tables) + an ONNX cross-encoder
   reranker that scores a bounded candidate prefix locally — no
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
│  │  fitz ...   │  │  import ... │  │  /answer    │                          │
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
│  - Deterministic planner + managed Qwen semantic keywords                   │
│  - Optional query-intelligence provider                                     │
│  - Router: symbol search · section search · table metadata / rows           │
│  - Expander (import graph, entity links, same-file refs, hierarchy)         │
│  - ONNX cross-encoder reranker (gte-reranker-modernbert-base)               │
│  - Optional synthesizer (chat call that writes the answer)                  │
│  - Governance (pyrrho → SUFFICIENT / DISPUTED / INSUFFICIENT)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────┐
│  LLM / ONNX         │  │  Storage (SQLite)   │  │  Ingestion Pipeline     │
├─────────────────────┤  ├─────────────────────┤  ├─────────────────────────┤
│  Qwen query terms + │  │  WAL + FTS5         │  │  Parse (CPU / Docling)  │
│  optional background│  │  one .db per        │  │  typed units: symbols,  │
│  work; endpoint chat│  │  collection         │  │  sections, tables       │
│  is explicit        │  │  bm25() ranking     │  │  index first, enrich    │
│                     │  │  json_each, json1   │  │  independently          │
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

| Layer | May import from |
| --- | --- |
| `core/` | `core/` |
| `encoders/` | `encoders/` |
| `ingestion/` | `core/`, `ingestion/` |
| `storage/` | `core/`, `storage/` |
| `retrieval/` | `core/`, `retrieval/`, `storage/` |
| `llm/` | `core/`, `encoders/`, `llm/` |
| `governance/` | `core/`, `encoders/`, `governance/` |
| `tabular/` | `core/`, `llm/`, `storage/`, `tabular/` |
| `config/` | `config/`, `core/` |
| `engines/` | `config/`, `core/`, `engines/`, `governance/`, `ingestion/`, `llm/`, `retrieval/`, `storage/`, `tabular/` |
| `api/`, `cli/`, `runtime/`, `sdk/`, `services/`, `tools/` | unrestricted orchestration layers |

---

## Data Flow

### Query

Retrieval runs as a broad recall → rerank → compile → deliver → Pyrrho
pipeline. A `RetrievalPass` is retrieve → fuse → rerank → read;
contract-driven evidence closure may issue bounded follow-up retrieval before
the fixed delivery set is submitted to Pyrrho.

```
1  Query prep      deterministic plan, explicit clauses, Qwen semantic keywords
                   optional query_intelligence rewrite/analyze/detect
2  Broad recall    symbol / section / table BM25 and intent fanout
3  Fuse            merge across strategies and deduplicate
4  Rerank          ONNX cross-encoder (gte-reranker-modernbert-base)
5  Read            fetch content for surviving addresses
6  Closure         issue bounded follow-up retrieval for unresolved obligations
7  Compile         enforce query-shape evidence obligations
8  Deliver         select a fixed top_k/top_read evidence set
9  Pyrrho          one authoritative decision over the delivered set
10 Record          optional RetrievalRun snapshots this same execution
11 Synthesize      optional chat call writes an Answer from governed evidence
```

`RetrievalRun` is built from the canonical governed result. Trace mode does not
invoke a diagnostic copy of the pipeline. Pyrrho replay operates only on
the frozen delivered evidence in a content-bearing record;
it does not claim to reproduce retrieval against a mutable collection. See
[Retrieval Execution Records](RETRIEVAL_RUNS.md).

### Ingestion

```
Files → Register manifest
      → Parse (CPU parser by default, Docling/vision/OCR optional,
        tree-sitter/AST for code, native table parsers)
      → Store typed units (symbols, sections, tables) in SQLite + FTS5
      → Resolve imports
      → Searchable source index ready
      → Optional background enrichment:
        entity graph + hierarchy
        demand summaries only for files surfaced by queries
```

---

## Chat Provider Model

The LLM layer has one managed local Qwen runtime for semantic query terms and
optional background work, plus one canonical optional endpoint provider —
**`endpoint`**:

| Spec                                   | Resolves to                                              |
| -------------------------------------- | -------------------------------------------------------- |
| `onnx/qwen3-0.6b`                     | managed Qwen3 0.6B ONNX GenAI generation on CPU          |
| `endpoint/<model>`                    | model plus `chat_base_url` and optional API-key env       |
| `openai/<model>`                       | endpoint pointing at `https://api.openai.com/v1`         |
| `azure_openai/<deployment>`            | endpoint with Azure deployment URL                       |
| `enterprise/<provider>/<model>`        | endpoint + M2M / mTLS / custom-CA auth                   |

Provider specs must resolve to the OpenAI-compatible chat protocol. Local
servers such as Ollama are configured through `endpoint` plus their `/v1`
base URL.

The **`OnnxReranker`** is a separate INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default). It uses Fitz-Sage's
`OnnxEncoderBackend`; Pyrrho uses a dedicated managed adapter for its validated
multi-head model contract.
The reranker scores a profile-aware 24, 32, or 48 `(query, candidate)` pairs
locally with no external LLM call.
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

Sections and symbols have FTS5 external-content indexes. Native table rows use
a separate row-value FTS5 index; table metadata, import edges, and entity links
use ordinary SQLite tables and indexes. Store search methods sign-flip FTS5's
negative `bm25()` value so downstream consumers treat higher as better.

See [features/platform/unified-storage.md](features/platform/unified-storage.md)
for schema and runtime details.

---

## Feature Control

Optional endpoint-backed features are controlled by **provider presence**, not
boolean flags. Query-time Qwen semantic expansion is standard engine behavior.
Background enrichment starts independently after the source index is ready and
does not have a public provider knob.

```yaml
# ENABLED — a provider is named
rerank: onnx
governance: pyrrho
synthesizer: endpoint/qwen2.5-7b-instruct
chat_base_url: http://localhost:8080/v1

# DISABLED — omit the key (or set null)
# synthesizer: null → no answer generation
```

---

## Core Types

```python
@dataclass
class Query:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    mode: AnswerMode | None       # runtime: SUFFICIENT | DISPUTED | INSUFFICIENT
    provenance: list[Provenance]  # source attribution chain
    metadata: dict


@dataclass
class RetrievalRun:
    query: QueryExecution
    evidence: EvidencePack
    strategies: tuple[StrategyExecution, ...]
    candidate_stages: tuple[CandidateStage, ...]
    pyrrho: PyrrhoExecution
    ranked_evidence: tuple[FrozenEvidence, ...]
    pyrrho_evidence: tuple[FrozenEvidence, ...]
    environment: RunEnvironment
    schema_version: str
```

There is no public `Chunk` type in the retrieval path — KRAG uses
**typed units** (`Symbol`, `Section`, `TableSpec`) with structural
metadata, not fixed-size text windows.

---

## Configuration Layout

```
.fitz/
├── config.yaml              # engine config (role providers, retrieval knobs)
├── sqlite/                  # one .db per collection
│   ├── fitz_default.db
│   └── ...
└── collections/
    └── <collection>/
        └── manifest.json    # index and enrichment state
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

The bare `governance: pyrrho` value uses Pyrrho's accepted immutable default.
Local package directories and remote packages pinned to full commits are
available for advanced deployments.

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
│   ├── detection/       # deterministic modules + optional LLM parsing
│   ├── entity_graph/    # Entity-based linking
│   └── rewriter/        # optional query-intelligence rewrite types
├── llm/                 # Managed ONNX query/background work + optional endpoints
│   ├── providers/       # onnx_chat, endpoint, enterprise, onnx_reranker
│   ├── auth/            # ApiKeyAuth, M2MAuth, CompositeAuth
│   ├── config.py        # provider-spec → instance factory
│   └── client.py        # get_chat, ...
├── storage/             # SqliteConnectionManager (WAL, FTS5)
├── ingestion/           # built-in parsers, source discovery, and hashing
├── tabular/             # native table parsers and SqliteTableStore
├── integrations/        # thin bridge from retrieval results to Pyrrho
├── runtime/             # multi-engine orchestration
├── cli/                 # typer commands
├── api/                 # FastAPI app + routes
└── sdk/                 # stateful Python interface
```

---

## Design Principles

1. **Explicit over clever.** No magic. Read the config; know what happens.
2. **Managed local models.** Query expansion uses local Qwen ONNX; background
   Qwen work is optional, and endpoint chat roles remain explicit.
3. **Structure-first retrieval.** Parse code/docs into typed units at
   ingest; route to the right strategy at query time.
4. **No embeddings.** BM25 + KRAG routing + ONNX rerank is the retrieval
   backbone; there are no dense indexes or vector columns.
5. **Honest over helpful.** Mark evidence insufficient instead of hallucinating.
6. **Narrow extension points.** Provider and parser implementations are wired
   explicitly; source cleanup stays outside the package.
7. **Local-first.** SQLite and managed ONNX query/background work run locally
   after models are cached; explicitly selected endpoint and OCR servers remain
   optional deployment dependencies.
8. **KRAG provenance.** KRAG answers trace back to source addresses; custom
   engine protocols remain engine-defined.

---

## See Also

- [Unified Storage](features/platform/unified-storage.md) — SQLite + FTS5
- [Retrieval Pipeline](RETRIEVAL_PIPELINE.md) — query flow, fixed evidence delivery, and indexing states
- [PLUGINS.md](PLUGINS.md) — supported extension points
- [CONFIG.md](CONFIG.md) — configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) — feature-control architecture
- [INGESTION.md](INGESTION.md) — ingestion pipeline
- [CONSTRAINTS.md](CONSTRAINTS.md) — epistemic guardrails
- [features/platform/openai-compatible-endpoint.md](features/platform/openai-compatible-endpoint.md) — the canonical chat provider
