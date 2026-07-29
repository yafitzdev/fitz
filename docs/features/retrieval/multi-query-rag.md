# Multi-Query Retrieval

## Problem

A compound question can contain several independent retrieval obligations. A
single BM25 result list may be dominated by the easiest clause, leaving no
evidence for the others.

For example:

```text
What is the refund window, and who approves exceptions?
```

The policy duration and the approval owner may live in different documents.

## Default Behavior

The deterministic query planner recognizes explicit clause boundaries such as
separate question marks, semicolons, and a new question or command after
`and`/`but`. It keeps the original query and adds bounded clause-specific BM25
searches:

```text
Original: What is the refund window, and who approves exceptions?
Leg 1:    What is the refund window
Leg 2:    who approves exceptions
```

Ordinary conjunctions remain intact:

```text
What are the retention and deletion policies?
Compare React and Vue performance.
How do I install and configure the agent?
```

This is query-shape parsing, not semantic rewriting.

## Retrieval Flow

```text
original query
    |
    +-- deterministic explicit clauses
    |
    +-- optional query-intelligence decomposition
             |
             v
        BM25 per query leg
             |
             v
    merge duplicates and retain query-leg provenance
             |
             v
       fuse and rerank within fixed budgets
             |
             v
 preserve one hit from each successful explicit leg when the budget permits
```

The package does not create placeholder evidence for a leg that found nothing,
and it does not increase the configured result budget to force coverage.

## Optional Query Intelligence

Configure `query_intelligence:` when implicit, conversational, or structurally
messy compound questions need model-assisted decomposition. A valid
model-provided decomposition takes precedence. If the model returns no
decomposition, deterministic explicit clauses remain active alongside any
model rewrite.

```yaml
query_intelligence: endpoint/qwen2.5-7b-instruct
chat_base_url: http://localhost:8080/v1
```

Leave `query_intelligence: null` for the default local path.

## Boundaries

Fitz-Sage does not infer that differently written identifiers or domain terms
mean the same thing. It also does not clean, compress, or reinterpret raw logs
before retrieval. Applications must normalize source data and provide
domain-specific vocabulary outside Fitz-Sage.

The deterministic splitter intentionally handles only explicit syntax. It does
not guess how an implicit multi-topic sentence should be decomposed.

## Related

- [Semantic Query Keywords](query-expansion.md) - managed-Qwen recall-term
  expansion
- [Sparse Search](sparse-search.md) - FTS5 and `bm25()` used for each query leg
- [Reranking](reranking.md) - the bounded ONNX cross-encoder precision stage
