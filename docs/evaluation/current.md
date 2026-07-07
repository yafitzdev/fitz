<!-- docs/evaluation/current.md -->
# Current Measurement Plan

This page tracks what fitz-sage should measure for the current retrieval-first
architecture. Historical benchmark writeups remain in this directory as records,
but they are not the product contract.

## What Needs Measurement

| Area | Metric | Why it matters |
|---|---|---|
| Retrieval recall | Hit@K / nDCG over fixture corpora | Broad recall must surface the right typed units before reranking. |
| Reranking | Hit@1 / MRR after ONNX rerank | The reranker should move the answerable source near the top. |
| Governance cutoff | False-sufficient rate, insufficient recall, disputed recall | Pyrrho must avoid trusting weak or contradictory evidence. |
| Progressive indexing | time-to-first-evidence, query-ready latency, full-enrichment latency | The user experience depends on useful evidence before deep indexing completes. |
| Modality routing | accuracy by text/table/code/log/config route | Bad routing makes good retrieval strategies invisible. |
| Structured/table evidence | SQL correctness, aggregation completeness, unit/filter correctness | Table evidence has failure modes that text governance does not cover. |
| Code evidence | symbol hit rate, caller/callee coverage, test/doc conflict detection | Code answers need source-level sufficiency, not just topical relevance. |

## Current Baseline Expectations

- `fitz query` returns an `EvidencePack` with provenance and Pyrrho mode.
- Retrieval uses FTS5/BM25, typed-unit routing, structural expansion, and ONNX
  reranking.
- Pyrrho evaluates ranked evidence prefixes and may trigger a broader retrieval
  retry when its retrieval-control heads request more evidence.
- Optional answer synthesis is measured separately from evidence retrieval.

## Gaps To Fill

1. A maintained retrieval benchmark over the current KRAG fixture corpora.
2. A governance evaluation split by evidence modality: text, table, code, logs,
   and mixed evidence.
3. A progressive-indexing latency benchmark that reports both query-ready and
   fully-enriched milestones.
4. A public regression report format that can be updated without turning the
   docs into release notes.

## Related

- [Evidence Pack](../EVIDENCE_PACK.md)
- [Three-Stage Retrieval Strategy](../features/retrieval/three-stage-strategy.md)
- [Governance Modality Boundaries](../features/governance/modality-boundaries.md)
