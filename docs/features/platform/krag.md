# KRAG — Knowledge Routing Augmented Generation

KRAG is fitz-sage's retrieval engine. It rejects three assumptions that
most RAG stacks share — that documents should be chunked into fixed
windows, that embeddings are the right index, and that all content
types deserve the same search strategy — and replaces them with
**typed-unit retrieval over a structural index**.

This document explains why and how it works.

---

## The problems KRAG addresses

### Traditional RAG: dumb chunks

Traditional RAG follows a linear pipeline: **chunk documents → embed
chunks → similarity search → generate answer**. It works for simple
factual lookups but breaks down predictably:

1. **Chunks are dumb boundaries.** A 512-token window doesn't know
   where a function ends or a section begins. You get half a class
   definition in one chunk and the other half in the next.
2. **No structural awareness.** Chunks don't know that `file_a.py`
   imports `helper()` from `file_b.py`. There's no concept of "what
   calls this?" or "what depends on that?"
3. **Content types are flattened.** Code, prose, tables, figures all
   become identical text blobs. A SQL table chunked into text
   fragments loses its queryable structure entirely.
4. **Retrieval is one-shot.** One similarity search, one result set.
   If the answer spans sources or requires traversing relationships,
   traditional RAG gives you random fragments.

### Agentic RAG: pay-per-hop reasoning

Agentic RAG wraps an LLM agent around the retrieval loop. The agent
rewrites queries, decides which tools to call, and iterates. But it
trades latency, cost, and predictability for the missing structure:

1. **Latency.** Every "hop" is an LLM call. A 3-hop retrieval means
   3 round-trips before generation even starts.
2. **Cost.** Each reasoning step burns tokens. Agent retrieval can
   5–10× the cost of a single query.
3. **Unpredictable behaviour.** Same query, different hop counts —
   makes latency, cost, and quality hard to guarantee.
4. **Compensating with reasoning.** The agent doesn't *know* that
   `AuthService` depends on `TokenValidator`. It has to *figure it
   out* by searching, reading, and reasoning. Expensive work that
   could be a simple graph lookup if the structure existed.

### GraphRAG: LLM-hallucinated edges

GraphRAG builds a knowledge graph from documents by running LLM
entity/relationship extraction, then summarises via community
detection. It's strong for "what are the main themes?" but:

1. **Ingestion is expensive.** Thousands of LLM calls just to build
   the graph.
2. **Hallucinated edges.** The graph is as reliable as the LLM's
   reading comprehension — it can invent relations or miss real
   ones.
3. **No native code understanding.** GraphRAG treats `AuthService`
   as a text label, not as a class with imports, methods, and call
   sites.
4. **Community-detection overhead.** Hierarchical Leiden clusters
   are statistical, not semantic.

---

## KRAG: structure-first retrieval

**Extract structure at ingestion; use it at query time.** Instead of
compensating for missing structure with reasoning, KRAG builds the
structure into the index.

```
Traditional RAG:
  Document → [chunk] [chunk] [chunk] → embed → similarity → answer

Agentic RAG:
  Document → [chunk] [chunk] [chunk] → embed → agent(search → reason → re-search → reason) → answer

GraphRAG:
  Document → LLM entity/relation extraction → graph → communities → answer

KRAG:
  Document → [symbols] [sections] [tables] → FTS5 + structure → routed search
           → ONNX rerank → fixed evidence → Pyrrho v2 → EvidencePack
```

KRAG does not use vector similarity search. The index is structural and lexical:
typed units, FTS5/BM25, graph expansion, and ONNX cross-encoder reranking.

### Core ideas

**1. Addresses, not chunks.**

KRAG doesn't store text fragments. It stores **addresses** — pointers
to specific, meaningful units of content:

- **Symbols.** A function, class, or method — extracted by tree-sitter
  with qualified name (`module.ClassName.method`), line range,
  references to other symbols, and import relationships.
- **Sections.** A heading and its content — extracted with
  parent/child hierarchy, level, and title.
- **Tables.** A structured dataset — stored as native SQLite tables
  with auto-detected schema (`SqliteTableStore`).

Each address has a summary, keywords, named entities, and structural
metadata. You never search raw text — you search the structured index.

**2. Import graphs, not text search.**

When KRAG ingests a Python codebase it builds an import graph. Every
file's imports are tracked: what it imports, where from, and what
imports it. So:

- "What depends on `AuthService`?" → graph traversal, not text search.
- "What would break if I change this function?" → reverse-dep lookup.
- "Show me callers of `validate_token()`" → follow references.

Traditional RAG would text-search for "validate_token" and hope it
appears near import statements. KRAG walks the graph.

**3. Content-type routing.**

Different content types need different strategies. KRAG routes
queries automatically:

| Content   | Strategy                                       | Why                                                       |
| --------- | ---------------------------------------------- | --------------------------------------------------------- |
| Code      | Symbol search (name + BM25 over qualified name + summary) | Functions have names, types, and summaries — use all   |
| Documents | Section search (BM25 + hierarchy)              | Sections have titles, parents, children — use the tree    |
| Tables    | Metadata and row-value BM25; optional SQL in answer path | Tables are structured - retrieve table evidence first, compute only when synthesis is configured |

**4. Structural expansion.**

After finding relevant addresses, KRAG expands context using
structural knowledge:

- **Same-file references.** If A references B and C in the same file,
  include them.
- **Import-based expansion.** If the query is about `engine.py`,
  include key symbols from files it imports.
- **Entity linking.** If two symbols share a named entity (e.g.,
  both mention `AuthService`), link them.
- **Section hierarchy.** If a deeply nested section matches, include
  its parent for context.

This is deterministic graph traversal — zero LLM calls.

**5. ONNX cross-encoder, not embedding cosine.**

The final ranking step is a dedicated INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default). It scores
24, 32, or 48 `(query, candidate)` pairs with two batch-one CPU workers.
No external LLM call, no embedding model. It uses the same
ModernBERT family as the Pyrrho v2 governance classifier; both run from
pre-built ONNX graphs on CPU.

---

## Comparison

| Dimension                | Traditional RAG    | Agentic RAG          | GraphRAG                       | KRAG                             |
| ------------------------ | ------------------ | -------------------- | ------------------------------ | -------------------------------- |
| Retrieval unit           | Fixed-size chunk   | Fixed-size chunk     | Entity / community node        | Symbols, sections, tables        |
| Structure awareness      | None               | Reasoned per-query   | LLM-extracted graph            | Deterministic (AST + imports)    |
| Cross-file dependencies  | Text search        | Multi-hop agent      | Entity co-occurrence           | Import graph traversal           |
| Content-type handling    | All treated as text| Agent chooses tools  | All treated as text            | Routed by content type           |
| Ingestion cost           | Low (chunk only)   | Low (chunk only)     | Very high (LLM per doc)        | Medium (parse + summarise)       |
| Query latency            | Fast               | Slow (N LLM calls)   | Fast                           | Fast (BM25 + local ONNX rerank/govern) |
| Cost per query           | Low                | High                 | Low                            | Low (no endpoint call for evidence) |
| Predictability           | Deterministic      | Non-deterministic    | Deterministic                  | Deterministic                    |
| Graph accuracy           | n/a                | n/a                  | Probabilistic                  | Exact (parsed from source)       |
| Code understanding       | Line-split text    | Agent reads + reasons| Entity labels only             | Parsed AST + qualified names     |
| Table queries            | Chunked text       | Agent generates SQL  | Chunked text                   | Native SQL execution             |
| Best for                 | Simple doc Q&A     | Complex multi-source | Corpus-level themes            | Code + docs + tabular            |

---

## Where KRAG uses agent-style techniques

KRAG keeps endpoint-backed query intelligence opt-in and bounded:

- **Query rewriting** to resolve pronouns / context when optional
  `query_intelligence` is configured — one chat call, not an agent loop.
- **Detection-based routing** that classifies query intent (temporal,
  comparison, aggregation) deterministically by default and can be enhanced by
  the optional query-intelligence bus.

The pattern: use structure and local ONNX models for retrieval and governance;
use endpoint LLMs only for optional query intelligence or generated answers.

---

## Architecture

```
                    ┌───────────────────────────────────────────┐
                    │             INGESTION                     │
                    │                                           │
  Code files ──────►  tree-sitter ──► SymbolStore (+ imports)   │
  Documents ───────►  parser ──────► SectionStore (+ hierarchy) │
  CSV/Excel ───────►  schema detect ► SqliteTableStore (+ SQL)  │
                    │                                           │
                    │  All units get: summary, keywords,        │
                    │  entities, hierarchy summary              │
                    │  Indexed into per-collection SQLite       │
                    │  with FTS5 external-content tables        │
                    └───────────────────────────────────────────┘
                                      │
                                      ▼
                    ┌───────────────────────────────────────────┐
                    │             QUERY TIME                    │
                    │                                           │
  Query ──► Plan + Qwen keywords ──► Route                      │
                    │                             │             │
                    │              ┌──────────────┤             │
                    │              ▼              ▼             │
                    │         CodeSearch    SectionSearch       │
                    │        (name+BM25     (BM25+hierarchy     │
                    │         via FTS5)      via FTS5)          │
                    │              │              │             │
                    │              ▼              ▼             │
                    │           Merge + Expand                   │
                    │           (import graph, entities,        │
                    │            same-file refs, neighbors)     │
                    │              │                            │
                    │              ▼                            │
                    │           OnnxReranker (ONNX cross-encoder)│
                    │              │                            │
                    │              ▼                            │
                    │     Read + compile + Pyrrho decision      │
                    │           → EvidencePack                  │
                    └───────────────────────────────────────────┘
```

---

## Two entry points

KRAG exposes its pipeline three ways:

- **`evidence(query)`** — return a governed `EvidencePack` with ranked
  source items, Pyrrho probabilities, fixed-delivery metadata, timings, and indexing
  status. This is the `fitz retrieve` contract.
- **`retrieve(query)`** — run retrieval only and return the expanded
  sources as `list[ReadResult]`, without evidence packaging.
- **`answer(query)`** — optional synthesis from the retrieved/governed context
  when `synthesizer:` is configured.

See [ENGINES.md](../../ENGINES.md) for usage.

---

## Files

| Component                       | Path                                                             |
| ------------------------------- | ---------------------------------------------------------------- |
| Engine                          | `fitz_sage/engines/fitz_krag/engine.py`                          |
| Config                          | `fitz_sage/engines/fitz_krag/config/schema.py`                   |
| Query analyzer                  | `fitz_sage/engines/fitz_krag/query_analyzer.py`                  |
| Retrieval router                | `fitz_sage/engines/fitz_krag/retrieval/router.py`                |
| Code search                     | `fitz_sage/engines/fitz_krag/retrieval/strategies/code_search.py`|
| LLM-driven code search          | `fitz_sage/engines/fitz_krag/retrieval/strategies/llm_code_search.py` |
| Section search                  | `fitz_sage/engines/fitz_krag/retrieval/strategies/section_search.py` |
| Table search                    | `fitz_sage/engines/fitz_krag/retrieval/strategies/table_search.py` |
| Context expander                | `fitz_sage/engines/fitz_krag/retrieval/expander.py`              |
| ONNX reranker                   | `fitz_sage/engines/fitz_krag/retrieval/reranker.py` + `fitz_sage/llm/providers/onnx_reranker.py` |
| Evidence closure                | `fitz_sage/engines/fitz_krag/evidence_closure.py`                |
| Symbol store                    | `fitz_sage/engines/fitz_krag/ingestion/symbol_store.py`          |
| Section store                   | `fitz_sage/engines/fitz_krag/ingestion/section_store.py`         |
| Table store                     | `fitz_sage/engines/fitz_krag/ingestion/table_store.py`           |
| Import graph store              | `fitz_sage/engines/fitz_krag/ingestion/import_graph_store.py`    |
| Ingestion pipeline              | `fitz_sage/engines/fitz_krag/ingestion/pipeline.py`              |

---

## Related Features

- [Retrieval Pipeline](../../RETRIEVAL_PIPELINE.md) — bounded contract-driven evidence closure
- [Entity Graph](../retrieval/entity-graph.md) — entity-based linking across retrieval units
- [Hierarchical RAG](../ingestion/hierarchical-rag.md) — L1 / L2 summaries for corpus-level context
- [Sparse Search](../retrieval/sparse-search.md) — FTS5 + `bm25()` ranking
- [Reranking](../retrieval/reranking.md) — ONNX cross-encoder reranker
- [Unified Storage](unified-storage.md) — SQLite + FTS5 layer
