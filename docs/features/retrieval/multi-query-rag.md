# Multi-Query RAG

## Problem

Standard RAG takes a large query (e.g., a full test report + spec
excerpt + requirements doc) and uses it as one BM25 query. The TF
distribution gets flat, the relevant tokens drown in surrounding
context, and FTS5 returns mostly noise.

## Solution: optional query decomposition

When `query_intelligence:` is configured, the query-prep rewrite section can
decompose a compound query into focused sub-queries, each run as its own
targeted FTS5 search. Without query intelligence, the deterministic planner
runs the original query plus cheaper keyword/intent fanout.

## How it works

```
Query comes in
    │
    ▼
Optional query-intelligence rewrite section
    │
    ├─ single-topic query  → one FTS5 + bm25() search
    │
    └─ compound / multi-topic query
           │
           ▼
       is_compound = true  →  focused decomposed sub-queries
           │
           ▼
       FTS5 + bm25() per sub-query  →  Dedupe  →  Rerank  →  Return
```

## Key Design Decisions

1. **Optional endpoint-backed enhancement** - Enable with `query_intelligence:` when compound-query decomposition is worth the endpoint call.

2. **Batched with other query intelligence** - Decomposition is one section of the query-prep call; it adds no extra round-trip beyond that optional call.

3. **Content-based** - Triggers when the query covers multiple distinct topics or points — not by length.

4. **LLM handles extraction** - No regex or entity configuration. The configured query-intelligence model figures out what's important.

5. **Explicit failure semantics** - If `query_intelligence:` is configured, the provider must return valid query-prep JSON. Without `query_intelligence:`, retrieval uses deterministic planning plus keyword/intent fanout.

## Example

**Input (long test report):**
```
Test TC_CAN_001 failed with error 0x4F on CAN Bus module.
The test was checking timeout behavior and got "no response" after 500ms.
Expected: ACK within 100ms. Actual: Timeout.
...
```

**LLM extracts:**
```json
["TC_CAN_001 known issues", "CAN Bus timeout", "error 0x4F", "no response timeout"]
```

**Result:** 4 targeted searches instead of 1 diluted search. Better retrieval precision.

## Configuration

```yaml
query_intelligence: endpoint/qwen2.5-7b-instruct
chat_base_url: http://localhost:8080/v1
```

Leave `query_intelligence: null` for the default no-endpoint path.

## Benefits

| Standard RAG | Multi-Query RAG |
|--------------|-----------------|
| 1 diluted query | N targeted queries |
| Random chunks | Relevant chunks |
| LLM filters noise | LLM analyzes |
| ~30-50% precision | ~60-80% precision |

## Use Cases

- Test failure analysis (automotive, QA)
- Log analysis with structured data
- Support ticket routing with metadata
- Any scenario with long, structured input + unstructured knowledge base

## Related

- [Query Expansion](query-expansion.md) — rule-based synonym/acronym
  expansion; complements multi-query
- [Sparse Search (FTS5 + bm25)](sparse-search.md) — what each
  sub-query actually hits
- [Hierarchical RAG](../ingestion/hierarchical-rag.md) — summaries
  help when multi-query extracts high-level themes
- [Reranking](reranking.md) — the ONNX cross-encoder step that follows
  the merged sub-query results
