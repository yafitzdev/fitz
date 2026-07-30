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
| Pyrrho integration | Exact input/output identity, one decision per retrieval, replay parity | Fitz must not reinterpret or override Pyrrho. |
| Source indexing | cold query-ready files/second, failures, no-change re-point latency | `point()` must make the ordinary retrieval index available quickly and deterministically. |
| Background enrichment | entity/hierarchy completion latency and failure inventory | Optional model work must not be confused with source availability. |
| Modality routing | accuracy by text/table/code/log/config route | Bad routing makes good retrieval strategies invisible. |
| Structured/table evidence | SQL correctness, aggregation completeness, unit/filter correctness | Table evidence has failure modes that text governance does not cover. |
| Code evidence | symbol hit rate, caller/callee coverage, test/doc conflict detection | Code answers need source-level sufficiency, not just topical relevance. |

## Current Baseline Expectations

- `fitz retrieve` returns an `EvidencePack` with provenance and Pyrrho mode.
- Retrieval uses FTS5/BM25, typed-unit routing, structural expansion, and ONNX
  reranking.
- Fitz-Sage consumes Pyrrho's query-only intent and evidence-kind heads as
  retrieval signals. Pyrrho evaluates the final delivered evidence set and owns
  the final governance verdict.
- Optional answer synthesis is measured separately from evidence retrieval.

## Completed External Measurements

- The broad NFCorpus, FiQA, and SciFact ablation measures literal retrieval,
  managed Qwen expansion, and INT8 reranking over all 1,271 judged queries.
- The frozen ArguAna/Quora semantic holdout measures 240 queries across low,
  medium, and high lexical-overlap strata. It found no consistent low-overlap
  recall gain from the current Qwen path and a conclusive Quora regression.
- External NapierOne runs measure cold indexing, unsupported inputs,
  interruption recovery, and storage growth over real files.

## Gaps To Fill

1. A separate semantic-expansion development set for model, filtering, and
   fusion experiments. The frozen ArguAna/Quora holdout must remain
   evaluation-only and should be rerun after the planned managed-model
   replacement.
2. Cross-modality integration cases that preserve the exact accepted Pyrrho
   output without treating it as Fitz retrieval quality. Pyrrho owns
   false-sufficient, class-recall, and calibration evaluation.
3. Keep the query-ready ingestion benchmark representative across small files,
   long documents, code, tables, and explicitly selected rich parsers.
4. Measure and improve unchanged `point()` behavior for collections containing
   hundreds of thousands of tiny files.
5. A public regression report format that can be updated without turning the
   docs into release notes.

## Related

- [Evidence Pack](../EVIDENCE_PACK.md)
- [Three-Stage Retrieval Strategy](../features/retrieval/three-stage-strategy.md)
- [Governance Modality Boundaries](../features/governance/modality-boundaries.md)
- [BEIR Component Ablation](beir-component-ablation-2026-07-30.md)
- [BEIR Semantic Holdout](beir-semantic-holdout-2026-07-30.md)
