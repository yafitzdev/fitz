# docs/features/retrieval/

Deep-dive documentation for fitz-sage's retrieval intelligence
modules. All of these run automatically — none require configuration
to enable.

| File                       | Feature                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| `sparse-search.md`         | FTS5 + native `bm25()` over typed-unit stores                    |
| `reranking.md`             | LLM-based reranker (single chat call scoring candidates)         |
| `query-rewriting.md`       | LLM-based query reformulation (pronouns, typos, intent)          |
| `query-expansion.md`       | Rule-based synonym / acronym expansion                           |
| `multi-query-rag.md`       | Decomposes long queries into focused sub-queries                 |
| `multi-hop-reasoning.md`   | Iterative retrieval with bridge extraction                       |
| `keyword-vocabulary.md`    | Exact-match identifier vocabulary (TC-123, AuthService, …)       |
| `entity-graph.md`          | Entity-based linking across typed units                          |
| `comparison-queries.md`    | Side-by-side comparison query handling                           |
| `aggregation-queries.md`   | Detection and handling of aggregation queries                    |
| `temporal-queries.md`      | Date-aware retrieval and freshness handling                      |
| `freshness-authority.md`   | Source freshness and authority scoring                           |

What's *not* here anymore (removed in v0.12.0):

- **Hybrid search.** Embeddings + sparse fusion required a vector
  layer that was removed; retrieval is now pure FTS5 + structural
  routing + LLM rerank.
- **HyDE.** Hypothetical Document Embeddings was an
  embeddings-only technique.
- **Contextual embeddings.** Same — relied on the embedding stack.

For the implementation, see `fitz_sage/retrieval/` and
`fitz_sage/engines/fitz_krag/retrieval/`.
