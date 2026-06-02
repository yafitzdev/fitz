<!-- docs/README.md -->
# Documentation

Start here when you need the current fitz-sage product model instead of
historical implementation notes.

## Core Guides

| File | Contents |
|---|---|
| `features/retrieval/three-stage-strategy.md` | Retrieval strategy: broad recall -> ONNX rerank -> Pyrrho cutoff |
| `RETRIEVAL_PIPELINE.md` | End-to-end query and progressive indexing flow |
| `QUERY_UX.md` | One-command CLI user journey and background indexing behavior |
| `EVIDENCE_PACK.md` | Governed evidence response contract |
| `MANAGED_MODELS.md` | Local ONNX models, download behavior, and cache expectations |
| `CLI.md` | Command reference |
| `SDK.md` | Python SDK reference |
| `API.md` | REST API reference |
| `API_REFERENCE.md` | Core Python data models and engine protocols |

## System Guides

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | Package architecture and module boundaries |
| `INGESTION.md` | Parser, chunking, typed-unit ingestion, and enrichment |
| `ENRICHMENT.md` | Required managed-Qwen enrichment stages |
| `CONSTRAINTS.md` | Pyrrho governance and epistemic honesty |
| `CONFIG.md` | Runtime configuration |
| `FEATURE_CONTROL.md` | Provider-presence feature control |
| `TROUBLESHOOTING.md` | Common operational issues |

## Feature Areas

| Directory | Contents |
|---|---|
| `features/retrieval/` | Sparse search, query expansion, reranking, multi-hop, entity graph, freshness, temporal, aggregation, comparison |
| `features/governance/` | Epistemic honesty and governance evaluation |
| `features/ingestion/` | Code symbols, tabular routing, hierarchy |
| `features/platform/` | KRAG, progressive indexing, unified storage, endpoint integration |
| `evaluation/` | Current benchmark results |
