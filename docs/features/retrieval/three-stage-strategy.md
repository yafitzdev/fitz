# Three-Stage Retrieval Strategy

Fitz-Sage separates broad recall, neural precision, and governed delivery. The
separation makes failures attributable: a source can be absent from recall,
lost during ranking/reading, removed during compilation, or judged inadequate
by Pyrrho.

## Stage 1: Broad Recall

The query plan combines:

- literal terms and exact identifiers from the user query;
- deterministic temporal, comparison, aggregation, and modality shape;
- Pyrrho query-only PRE obligations;
- managed Qwen semantic query terms;
- optional endpoint-backed query intelligence;
- explicit multi-clause and query-shape variations.

The router searches eligible source surfaces:

| Surface | Recall mechanism |
|---|---|
| Document sections | SQLite FTS5 BM25 over title and original content |
| Code symbols | FTS5 BM25 plus case-insensitive name lookup |
| Native CSV/TSV tables | table name/columns, row-value BM25, bounded row plans |

All recall legs merge into one bounded pool with exact address deduplication.
Literal and expanded candidates intentionally compete: broad recall gives
alternate vocabulary room to enter rather than preserving the complete literal
tail.

There is no raw-file supplemental scan. `point()` completes the persisted source
index before retrieval starts.

## Stage 2: Rerank And Read

The INT8 ONNX cross-encoder scores a profile-aware candidate prefix. The full
recall pool remains available to concrete-row and evidence-contract rescue, but
neural work is bounded independently.

Surviving addresses are then read from stored original source. KRAG can add
bounded structural context from section hierarchy, code imports/references, and
optional entity or corpus hierarchy metadata.

If the first read set does not satisfy a mechanical evidence obligation,
evidence closure may issue bounded request-specific recall before compilation.
It skips a request when the requested physical modality is absent from the
collection and records that skip in the retrieval trace.

## Stage 3: Compile, Deliver, Govern

The evidence compiler applies query-shape obligations without inventing source
facts. It can retain explicit comparison sides, temporal scope, exact literal
anchors, requested modalities, and broad coverage roles when those candidates
exist.

A fixed `top_k`/`top_read` evidence set is then delivered once to Pyrrho:

```text
compiled source evidence
    -> fixed delivery set
    -> Pyrrho(query, exact evidence)
    -> SUFFICIENT / DISPUTED / INSUFFICIENT
    -> EvidencePack
```

Pyrrho owns logit decoding, thresholds, failure mode, and the final verdict.
Fitz-Sage does not add confidence floors, evidence-count overrides, or retry a
different evidence prefix after seeing the verdict.

## Optional Background Context

Entity links, L1/L2 hierarchy summaries, and demand summaries can improve later
context after background enrichment. They are not indexing prerequisites. A
collection with pending or failed enrichment uses the same source section,
symbol, and table indexes.

## Returned Evidence

`EvidencePack` reports:

- ranked source units and provenance;
- the runtime mode and exact Pyrrho metadata;
- query profile and planning ownership;
- retrieval, closure, compiler, and delivery metadata;
- source-index and enrichment status;
- stage timings.

`RetrievalRun` adds term origins, candidate stages, content hashes, model
fingerprints, and optional frozen content for Pyrrho-only replay.

## Boundaries

- Broad recall cannot bridge private aliases or invisible vocabulary reliably.
- Pointwise reranking can be weak on set-level and multi-document coverage.
- Fixed budgets can exclude a relevant source.
- Evidence closure is bounded and can only search physical modalities that
  exist.
- Pyrrho model quality and context length are governance boundaries, not Fitz
  retrieval heuristics.

## Related

- [Retrieval Pipeline](../../RETRIEVAL_PIPELINE.md)
- [Evidence Signals](evidence-signals.md)
- [Reranking](reranking.md)
- [Limitations](../../LIMITATIONS.md)
