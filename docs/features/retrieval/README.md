# docs/features/retrieval/

Deep-dive documentation for fitz-sage's retrieval intelligence modules. The
default pipeline is broad recall → ONNX rerank → Pyrrho cutoff. See
[Three-Stage Retrieval Strategy](three-stage-strategy.md) for the product model
and [Retrieval Pipeline](../../RETRIEVAL_PIPELINE.md) for the end-to-end flow.

| File                       | Feature                                                          |
| -------------------------- | ---------------------------------------------------------------- |
| `three-stage-strategy.md`  | Recall → rerank → Pyrrho strategy and how all tactics fit        |
| `sparse-search.md`         | FTS5 + native `bm25()` over typed-unit stores                    |
| `reranking.md`             | INT8 ONNX cross-encoder reranker (gte-reranker-modernbert-base)  |
| `query-rewriting.md`       | Optional query-intelligence reformulation (pronouns, typos, intent) |
| `query-expansion.md`       | Dictionary + managed-Qwen keyword expansion                      |
| `multi-query-rag.md`       | Optional decomposition for long compound queries                 |
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
  routing + ONNX cross-encoder rerank.
- **HyDE.** Hypothetical Document Embeddings was an
  embeddings-only technique.
- **Contextual embeddings.** Same — relied on the embedding stack.

For the implementation, see `fitz_sage/retrieval/` and
`fitz_sage/engines/fitz_krag/retrieval/`.
