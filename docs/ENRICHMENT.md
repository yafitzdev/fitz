<!-- docs/ENRICHMENT.md -->
# Enrichment

Required LLM-powered metadata added to ingested content. Enrichment is part of
the KRAG index contract: ingestion should produce keywords, entities, temporal
metadata, and hierarchy summaries before the collection is treated as ready.

---

## Overview

Enrichment adds AI-generated metadata to typed retrieval units (code symbols,
document sections) during ingestion. It runs as part of the KRAG ingestion
pipeline — no separate orchestrator, and no foreground query dependency.

```
┌─────────────────────────────────────────────────────────────────┐
│  KRAG Ingestion Pipeline (KragIngestPipeline)                   │
│  per file:  parse → summarize → enrich      corpus:  finalize   │
└─────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  enrich step                                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    KragEnricher                         │    │
│  │      One LLM call per batch (~15 symbols/sections)       │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │  Keywords          │  Entities          │  Temporal      │    │
│  │  exact-match IDs   │  named entities    │  dates /       │    │
│  │  (TC-1001, class   │  with types        │  versions /    │    │
│  │  names, ...)       │  ({name, type})    │  refs          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │       Hierarchy summaries (pipeline-built)               │    │
│  │  L1: one group summary per document file                 │    │
│  │  L2: one corpus summary rolled up from the L1 summaries  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Key features:**
- **Keywords** — exact-match identifiers (TC-1001, JIRA-123, `AuthService`)
- **Entities** — named entity extraction (classes, people, technologies)
- **Temporal metadata** — dates, version numbers, time references found in the text
- **Hierarchy** — L1 (per-file) and L2 (corpus) summaries for analytical queries

Keyword/entity/temporal extraction uses `enricher:`. L1/L2 hierarchy summaries
use `summarizer:`. The default profile is `qwen3.5-0.8b@Q4_K_M` behind a local
OpenAI-compatible endpoint. If either provider is missing or unreachable,
ingestion fails closed instead of silently storing an under-enriched index.

---

## Architecture: KragEnricher

`KragEnricher` (`fitz_sage/engines/fitz_krag/ingestion/enricher.py`) extracts keywords, entities, and temporal metadata in a single LLM call per batch of symbols or sections.

### How it works

1. **Batching** — symbols and sections are processed in batches (`summary_batch_size`, default 15)
2. **Single LLM call** — one call extracts keywords + entities + temporal references for the whole batch
3. **JSON response** — the LLM returns a JSON array, one object per item
4. **In-place enrichment** — each item dict is updated with `keywords`, `entities`, and (when present) `metadata["temporal"]`

```python
# One LLM call returns one object per symbol/section:
[
  {
    "keywords": ["AuthService", "OAuth2", "JWT_TOKEN"],
    "entities": [{"name": "AuthService", "type": "class"}, ...],
    "temporal": {"dates": ["2024-03"], "versions": ["v2.3"], "refs": ["latest"]}
  },
  # ... for each item in the batch
]
```

`KragEnricher.enrich_symbols()` enriches code symbols; `enrich_sections()` enriches document sections. Code symbols are described to the LLM by name + kind; sections by title + summary/content.

---

## What it extracts

### Keywords

Exact-match identifiers — function names, class names, technical terms, IDs, abbreviations. These back exact-identifier matching at query time.

**Examples:** `TC-1001`, `JIRA-4521`, `v2.0.1`, `AuthService`, `MAX_RETRIES`, `/api/v2/users`

**Stored on:** each symbol's / section's `keywords` field (persisted to the symbol / section store).

Exact-identifier lookup at query time is delivered by SQLite FTS5 + native `bm25()` — see [Sparse Search](features/retrieval/sparse-search.md). There is no separate vocabulary store.

### Entities

Named entities and domain concepts, each tagged with a type.

**Entity types:** `class`, `function`, `person`, `organization`, `technology`, `concept`

**Stored on:** each unit's `entities` field. During the enrich step the pipeline also feeds these into the [Entity Graph](features/retrieval/entity-graph.md) (`EntityGraphStore`) for related-unit discovery at query time.

### Temporal metadata

Dates, version numbers, and relative time references found in the unit text.

**Stored on:** `metadata["temporal"]` — `{dates, versions, refs}`.

---

## Hierarchy

L1 and L2 summaries for analytical queries. Built by `KragIngestPipeline`
itself when `summarizer:` is configured.

### The problem

Standard RAG struggles with analytical questions:

```
Q: "What are the main themes in these documents?"
Standard RAG: Returns random individual sections (not useful)
```

### The solution

The pipeline builds two summary layers for document content:

```
┌─────────────────────────────────────────────────────────────────┐
│  Level 2: Corpus Summary                                        │
│  "Across these documents, the main themes are: API design,      │
│   security best practices, and deployment patterns."            │
│  Stored as a synthetic retrievable section ("Corpus Overview")  │
└─────────────────────────────────────────────────────────────────┘
                              ▲   rolled up from L1
         ┌────────────────────┼────────────────────┐
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Level 1: file   │  │ Level 1: file   │  │ Level 1: file   │
│ group summary   │  │ group summary   │  │ group summary   │
│ stored on each  │  │ stored on each  │  │ stored on each  │
│ section's       │  │ section's       │  │ section's       │
│ metadata        │  │ metadata        │  │ metadata        │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

- **L1 — per-file group summary.** During `enrich_file`, the pipeline summarizes a document file's sections into one 2-3 sentence group summary, stored on each section's `metadata["hierarchy_summary"]`. Code symbols carry their own machine-readable structure (imports, AST), so they get no hierarchy summary.
- **L2 — corpus summary.** During the corpus `finalize` step, the pipeline rolls all L1 summaries up into a single 3-5 sentence corpus summary, stored as a synthetic retrievable section titled "Corpus Overview".

### Query behavior

| Query Type | Retrieved |
|------------|-----------|
| "What are the trends?" | L2 corpus summary scores high on abstract phrasing |
| "Explain the auth module" | the module's sections (L0), matched on their text |
| "How does validateToken work?" | the granular L0 sections |

No special query syntax needed — the L2 summary is an ordinary retrievable section, matched via FTS5/BM25 like any other.

---

## CLI Usage

Source-backed retrieval waits for required indexing before retrieving evidence:

```bash
fitz retrieve "your question" --source ./docs
```

The default local enrichment runtime is llama.cpp's `llama-server`:

```bash
llama-server -hf bartowski/Qwen_Qwen3.5-0.8B-GGUF:Q4_K_M \
  --alias qwen3.5-0.8b@Q4_K_M \
  --host 127.0.0.1 --port 8080
```

Config:

```yaml
enricher: endpoint/qwen3.5-0.8b@Q4_K_M
summarizer: endpoint/qwen3.5-0.8b@Q4_K_M
chat_base_url: http://127.0.0.1:8080/v1
```

If no inference engine is running, first-run setup writes this config and shows
the `llama-server` command. A source-backed retrieval then stops at indexing
with an actionable error until the local runtime is started.

---

## Cost Analysis

Batching keeps enrichment cheap. `KragEnricher`
extracts keywords + entities + temporal for ~15 symbols/sections per LLM call.

### Per batch (~15 units)

| Component | Tokens |
|-----------|--------|
| Prompt overhead | ~500 |
| Unit content (15 × ~400) | ~6,000 |
| **Total input** | **~6,500** |
| Response (15 × ~100) | ~1,500 |
| **Total output** | **~1,500** |

### Cost per model

| Model | Per Batch | Per Unit | 1000 Units |
|-------|-----------|----------|------------|
| Claude 3.5 Haiku | $0.011 | $0.0007 | **$0.74** |
| GPT-4o-mini | $0.002 | $0.0001 | **$0.13** |

L1/L2 hierarchy summaries add a small number of additional calls (one per
document file plus one corpus call).

---

## Configuration

Provider specs bind the required enrichment stages:

| Key | Controls |
|------|----------|
| `enricher: <provider/model>` | Keyword / entity / temporal extraction (`KragEnricher`) |
| `summarizer: <provider/model>` | L1 (per-file) and L2 (corpus) hierarchy summaries |

Small CPU-local profile:

```yaml
chat_base_url: http://127.0.0.1:8080/v1
enricher: endpoint/qwen3.5-0.8b@Q4_K_M
summarizer: endpoint/qwen3.5-0.8b@Q4_K_M
summary_batch_size: 15
```

`summary_batch_size` (default 15) sets the LLM batch size for both summarization and enrichment.

---

## Key Files

| File | Purpose |
|------|---------|
| `fitz_sage/engines/fitz_krag/ingestion/enricher.py` | `KragEnricher` — batched keyword/entity/temporal extraction |
| `fitz_sage/engines/fitz_krag/ingestion/pipeline.py` | `KragIngestPipeline` — drives enrich + builds L1/L2 summaries |
| `fitz_sage/engines/fitz_krag/config/schema.py` | `enricher` / `summarizer` provider specs |
| `fitz_sage/retrieval/entity_graph/` | `EntityGraphStore` — populated from extracted entities |

---

## See Also

- [INGESTION.md](INGESTION.md) - Full ingestion pipeline
- [CONFIG.md](CONFIG.md) - Configuration reference
- [CONSTRAINTS.md](CONSTRAINTS.md) - Epistemic guardrails
