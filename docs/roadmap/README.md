# docs/roadmap/README.md
# fitz-sage Roadmap

---

## Next Major: Corpus Intelligence (v0.11.0)

**[Corpus Intelligence — Self-Aware RAG with Actionable Quality Signals](./corpus-intelligence.md)**

Surface fitz-sage's hidden intelligence to developers. Actionable ABSTAIN (explains gaps, suggests documents to add), confidence scores, answer explanations, and corpus health reports. Zero new LLM calls — exposes signals already computed by governance and entity graph.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Actionable ABSTAIN (gap analysis in answer text) | **Done** |
| 2 | Confidence score on every Answer | Proposed |
| 3 | Answer explanation from constraint metadata | Proposed |
| 4 | Corpus health report (`fitz_sage.health()`) | Proposed |
| 5 | ABSTAIN-driven ingestion suggestions | Proposed |

---

## Next: Query Intelligence Pipeline

**[Rewrite-First with Batched Classification](./query-intelligence-pipeline.md)**

Reorder query preprocessing: rewrite first, then batch analysis + detection on the cleaned query. Reduces local LLM calls from 3 to 2, improves classification accuracy. Phase 2 adds extended signals (specificity, domain, multi-hop) to replace hard-coded retrieval gates.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Rewrite-first pipeline + batched analysis/detection | In Progress |
| 2 | Extended classification signals (specificity, domain, multi-hop) | Proposed |

---

## Next: Query Expansion Without Embeddings (HyDE replacement)

**[Vocabulary-Grounded Query Expansion](./query-expansion-without-embeddings.md)**

Fill the BM25 vocabulary-mismatch gap that HyDE used to cover, without bringing embeddings back. Two approaches: query-time LLM keyword expansion (Query2Doc / generative QE — well-studied, costs 1 extra LLM call per query) and a vocabulary-grounded shortcut that reuses the enrichment bus — `KragEnricher` already builds per-collection keyword variations at ingest time, and `KeywordMatcher.find_in_query()` already returns them. At query time, just OR the matched variations into FTS5. Zero query-time LLM cost. Hybrid Approach 3 falls back to LLM expansion only on vocab-miss.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | `QueryExpander` over existing `VocabularyStore` / `KeywordMatcher` (Approach 2) | Proposed |
| 2 | BEIR A/B vs. baseline BM25 (gated on Task 12 — restore evaluation subpackage) | Proposed |
| 3 | LLM-expansion fallback for vocab-miss queries (Approach 3 hybrid) | Proposed |

---

## Future: KRAG Agent

**[KRAG Agent — Retrieval-as-Tools with Epistemic Self-Verification](./krag-agent.md)**

Transform the pipeline into an autonomous agent using retrieval strategies as composable tools. Better suited if fitz-sage expands beyond library into research/investigation use cases.

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Tool definitions & agent core loop | Proposed |
| 2 | Self-verification tool (governance as in-loop primitive) | Proposed |
| 3 | Cross-collection federation | Proposed |
| 4 | Reasoning trace & observability | Proposed |
| 5 | Auto mode & confidence routing (pipeline vs agent) | Proposed |

---

## Ingestion Quality (post-v0.9.0)

PDF parsing is solved (pypdfium2 fast path + Docling OCR fallback with caching).
Remaining domains:

| # | Problem | File | Impact | Effort |
|---|---------|------|--------|--------|
| 1 | [DOCX/PPTX have no structure-aware chunking](./01-docx-pptx-chunking.md) | `ingestion/chunking/` | High | Medium |
| 2 | [Non-Python code silently loses all symbols](./02-tree-sitter-failures.md) | `progressive/worker.py` | High | Low |
| 3 | [CSV rows invisible to semantic search](./03-table-vector-gap.md) | `chunking/plugins/table.py` | Medium | Medium |
| 4 | [Python syntax errors = zero extraction](./04-python-syntax-fallback.md) | `strategies/python_code.py` | Medium | Low |

Each file is self-contained: problem, evidence, proposed fix, affected files.
