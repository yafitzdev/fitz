<!-- docs/RETRIEVAL_PIPELINE.md -->
# Retrieval Pipeline

fitz-sage is retrieval-first. The default product surface returns a governed
`EvidencePack`: ranked source units, Pyrrho metadata, indexing status, timings,
and enough provenance for another application to decide what to do next.
Generated answers are optional and live behind `fitz answer` / `fitz_sage.query()`.

For the retrieval strategy itself, see
[Three-Stage Retrieval Strategy](features/retrieval/three-stage-strategy.md).
For the no-flags CLI journey, see [Query UX](QUERY_UX.md). For the returned
object shape, see [Evidence Pack](EVIDENCE_PACK.md).

---

## User Journey

The intended CLI journey is one command:

```bash
fitz query "Which documents are relevant?"
```

When run from a document folder, this command:

1. Registers the current directory as the source.
2. Derives the collection name from the folder name.
3. Parses enough structure to make the corpus searchable.
4. Returns governed evidence.
5. Starts a detached indexing daemon when Qwen enrichment is still pending.

Use `fitz retrieve` only when you need advanced evidence controls such as
`--format json` or `--top-k`.

---

## End-to-End Flow

```mermaid
flowchart TD
    A["fitz query / fitz retrieve"] --> B{"Source supplied?"}
    B -->|"yes"| C["Register source into collection"]
    B -->|"no, no collection"| D["Use current directory as source"]
    B -->|"collection supplied"| E["Load existing collection"]
    D --> C
    C --> F["Build/update manifest"]
    F --> G["Parse files into searchable units"]
    G --> H["Search surface ready"]
    E --> I["Run retrieval pipeline"]
    H --> I
    I --> J["Return EvidencePack"]
    J --> K{"Deep enrichment complete?"}
    K -->|"yes"| L["Exit"]
    K -->|"no"| M["Spawn index-daemon"]
    M --> L
```

The foreground command waits for the search surface, not for full enrichment.
That gives the user a fast first evidence pack while the daemon keeps building
the richer index in the background.

---

## Query Pipeline

```mermaid
flowchart TD
    Q["User query"] --> C["Query profiling"]
    C --> P["Query prep"]
    P --> R["Broad recall"]
    R --> X["Cross-strategy fusion"]
    X --> K["ONNX reranker"]
    K --> G["Pyrrho cutoff loop"]
    G --> E["EvidencePack"]

    C --> C1["deterministic signals / optional trained query heads"]

    P --> P1["Deterministic terms, query type, intent detection"]
    P --> P2["Managed Qwen semantic keywords"]
    P --> P3["Optional query_intelligence rewrite / analysis / detection"]

    R --> R1["Section BM25 over FTS5"]
    R --> R2["Code symbol BM25 / name search"]
    R --> R3["Table metadata search"]
    R --> R4["Unindexed scan for files not query-ready"]

    G --> G1["Shape-aware evidence prefix"]
    G1 --> G2["Evaluate query + top 1"]
    G2 --> G3{"SUFFICIENT?"}
    G3 -->|"yes"| E
    G3 -->|"no"| G4["Evaluate query + top 2"]
    G4 --> G5{"Enough evidence or max cutoff?"}
    G5 -->|"continue"| G4
    G5 -->|"stop"| E
```

### Stage 1: Broad Recall

Broad recall is intentionally permissive. It uses real query terms, dictionary
synonyms/acronyms, managed Qwen semantic keywords, and intent fanout for
comparison, temporal, aggregation, and freshness queries. False positives are
acceptable because the reranker and governance cutoff handle precision.
The default Pyrrho v2 package is evidence-conditioned and does not add
pre-retrieval query heads. Query profiling comes from deterministic signals,
managed Qwen semantic keywords, and optional query-intelligence providers.
Explicit Pyrrho packages that actually train query heads may still contribute
query metadata, but v2 does not project those labels.

Primary stores:

| Store | Retrieval unit | Search surface |
|-------|----------------|----------------|
| `SectionStore` | document sections and synthetic summaries | SQLite FTS5 + `bm25()` |
| `SymbolStore` | code symbols | name search + SQLite FTS5 + `bm25()` |
| `TableStore` | table metadata | table name/schema search |
| Manifest scan | files not yet query-ready | path/heading/symbol BM25, optional file-selection LLM if configured |

### Stage 2: Rerank

The ONNX cross-encoder reranker scores `(query, candidate)` pairs after broad
recall. It is the precision stage. The default backend is
`Alibaba-NLP/gte-reranker-modernbert-base` through `onnxruntime`.

### Stage 3: Pyrrho Cutoff

Pyrrho does not answer the query. It decides whether the ranked evidence prefix
is sufficient.

```mermaid
flowchart TD
    A["Reranked candidates"] --> B["Take prefix of size 1"]
    B --> C["Pyrrho(query, prefix)"]
    C --> D{"Verdict"}
    D -->|"SUFFICIENT"| T["Stop: enough evidence"]
    D -->|"INSUFFICIENT"| E{"Reached max cutoff?"}
    D -->|"DISPUTED"| F{"Dispute stable enough?"}
    F -->|"yes"| U["Stop: return disputed evidence"]
    F -->|"no"| N["Add next document"]
    E -->|"no"| N
    E -->|"yes"| A0["Stop: insufficient"]
    N --> C
```

The default cutoff inspects at most the top 10 evidence items, or fewer when
the caller requested a smaller `top_k`.

---

## Indexing State Machine

```mermaid
stateDiagram-v2
    [*] --> REGISTERED
    REGISTERED --> PARSED: parse_file
    PARSED --> KEYWORDED: Qwen keyword_file
    KEYWORDED --> QUERY_READY
    QUERY_READY --> ENTITY_LINKED: link_entities_file
    ENTITY_LINKED --> HIERARCHY_READY: build_hierarchy_file
    HIERARCHY_READY --> ENRICHED
    ENRICHED --> SUMMARIZED: demand summarize_file
```

| State | User impact |
|-------|-------------|
| `REGISTERED` | File is known but not searchable yet. |
| `PARSED` | Raw content, symbols, sections, and tables are searchable. This is the foreground gate. |
| `QUERY_READY` | Managed Qwen keyword enrichment is complete for that file. |
| `ENTITY_LINKED` | Entity graph links are available. |
| `HIERARCHY_READY` | L1 hierarchy summaries are available. |
| `ENRICHED` | Required deep enrichment is complete. |
| `SUMMARIZED` | Demand summary exists because a query surfaced this file. |

`complete` in `indexing_status` means the query-ready keyword phase is done.
`fully_enriched` means entity/hierarchy enrichment is also done.

---

## Query Cases

### Case 1: No collection exists

`fitz query "..."` registers the current directory, parses it, retrieves a
best-effort evidence pack, and starts the daemon if enrichment remains.

### Case 2: Source registered, search surface not ready

The CLI waits while parsing runs. It may show `Parsing documents... N/M`.
Retrieval starts once parsed units are searchable.

### Case 3: Search surface ready, enrichment still pending

Retrieval uses parsed sections, symbols, tables, and the unindexed scan for any
files not yet query-ready. The output may show `Indexing pending`,
`Enrichment pending`, or `Deep enrichment pending`. The daemon continues Qwen
keywords, entities, hierarchy, and demand summaries.

The supplemental scan only runs when the manifest still has files below
query-ready. Fully query-ready collections do not print scan progress or touch
disk fallback.

### Case 4: Index is complete

Retrieval uses the fully populated stores. No daemon is spawned, and later
queries should be faster because Qwen keyword/entity/hierarchy enrichment has
already run.

### Case 5: Optional answer synthesis

`fitz answer` runs the same retrieval pipeline, then sends governed context to
the configured synthesizer. This is separate from the retrieval package default.

---

## Strategy Roles

| Strategy | Role |
|----------|------|
| Sparse BM25 / keyword vocabulary | Broad recall backbone. |
| Managed Qwen semantic query keywords | Broad recall expansion in the default no-endpoint path. |
| Optional Pyrrho query heads | Pre-retrieval signals only when the configured Pyrrho package actually trains query heads. |
| Dictionary query expansion | Fast synonyms/acronyms, no LLM call. |
| Query rewriting | Optional `query_intelligence` enhancement for conversational context or ambiguous phrasing. |
| Multi-query decomposition | Optional `query_intelligence` enhancement for compound questions. |
| Comparison / temporal / aggregation / freshness detection | Deterministic default signals, optionally improved by query intelligence. |
| Entity graph | Context expansion after full enrichment. |
| Hierarchical summaries | Fully indexed recall for broad analytical questions. |
| Unindexed scan | Temporary bridge while files are not query-ready. |
| ONNX reranker | Precision stage before governance. |
| Pyrrho | Mandatory sufficiency, dispute, and insufficiency cutoff for evidence packs. Comparison-shaped metric queries seed cutoff with direct metric/table evidence before Pyrrho can stop. |
| Multi-hop | Bounded bridge retrieval when the first pass is still insufficient and the answer appears one hop away. |

Synthetic corpus summaries are not normal section hits. They are schema-versioned,
deleted before regeneration, excluded from ordinary BM25, and injected only when
the query contract/profile calls for a representative corpus overview.

---

## Model Responsibilities

| Model/runtime | Required? | Used for |
|---------------|-----------|----------|
| Managed Qwen3 0.6B ONNX GenAI | yes | ingestion keywords/entities/hierarchy and default semantic query keywords |
| ONNX reranker | default | candidate precision after broad recall |
| Pyrrho v2 nano g1 | default product governance | native evidence verdict, failure mode, retrieval intents, and evidence-kind metadata |
| OpenAI-compatible endpoint | optional | answer synthesis, optional query intelligence, optional vision parser |

No dense embedding model and no vector database are used.
