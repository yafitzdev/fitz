<!-- docs/README.md -->
# Documentation

Start here when you need the current fitz-sage product model instead of
historical implementation notes.

## Core Guides

| File | Contents |
|---|---|
| `features/retrieval/three-stage-strategy.md` | Retrieval strategy: broad recall -> ONNX rerank -> fixed evidence -> Pyrrho |
| `features/retrieval/evidence-signals.md` | Pre-retrieval planning and post-retrieval evidence signals |
| `RETRIEVAL_PIPELINE.md` | End-to-end query and source-index lifecycle |
| `QUERY_UX.md` | One-command CLI journey and optional background enrichment |
| `EVIDENCE_PACK.md` | Governed evidence response contract |
| `MANAGED_MODELS.md` | Local ONNX models, download behavior, and cache expectations |
| `BENCHMARK.md` | Current benchmark results, methodology, timings, and interpretation |
| `PRODUCTION_READINESS.md` | Package responsibilities, release gates, and measured boundaries |
| `LIMITATIONS.md` | Input contract, user responsibilities, and measured product boundaries |
| `features/governance/modality-boundaries.md` | Pyrrho governance boundaries for text, tables, code, logs, and config |
| `CLI.md` | Command reference |
| `SDK.md` | Python SDK reference |
| `API.md` | REST API reference |
| `API_REFERENCE.md` | Core Python data models and engine protocols |

## System Guides

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | Package architecture and module boundaries |
| `INGESTION.md` | Parsers, typed-unit source indexing, and enrichment lifecycle |
| `ENRICHMENT.md` | Optional background entity and hierarchy work |
| `CONSTRAINTS.md` | Pyrrho governance and epistemic honesty |
| `CONFIG.md` | Runtime configuration |
| `FEATURE_CONTROL.md` | Provider-presence feature control |
| `TROUBLESHOOTING.md` | Common operational issues |
| `evaluation/current.md` | Current measurement plan |

## Feature Areas

| Directory | Contents |
|---|---|
| `features/retrieval/` | Sparse search, semantic keywords, reranking, evidence closure, entity graph, freshness, temporal, aggregation, comparison |
| `features/governance/` | Epistemic honesty and Pyrrho governance |
| `features/ingestion/` | Code symbols, tabular routing, hierarchy |
| `features/platform/` | KRAG, source indexing, unified storage, endpoint integration |
