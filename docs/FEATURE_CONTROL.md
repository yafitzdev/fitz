# Feature Control Architecture

How provider-backed features (VLM in the parser, optional synthesis, optional
query intelligence, and advanced model swaps) are declared in fitz-sage.

---

## Design Philosophy

fitz-sage uses a **provider-presence pattern**:

- **Config declares WHICH** provider/model to use.
- **Provider presence determines IF** optional endpoint-backed features run.
- **No `enabled: true / false` flags.** Setting a provider enables
  the feature; omitting it (or setting `null`) skips that step.
- **Retrieval intelligence is baked in.** Managed Qwen semantic query terms,
  broad recall, ONNX reranking, and Pyrrho governance are standard. Optional
  background enrichment is independent of source-index readiness.

This keeps the config declarative and avoids boolean flags that
can drift out of sync with the actual provider config.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  CONFIG (.fitz/config.yaml in the current workspace)            │
│  Declares WHICH provider/model to use                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  vision: endpoint/gpt-4o         rerank: onnx              │
│  parser: docling_vision          collection: default            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PROVIDER PRESENCE determines IF the feature is used            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  VLM (controlled by parser plugin):                             │
│    parser: cpu              → No VLM (fast default parser)      │
│    parser: docling          → No VLM (figures become "[Figure]")│
│    parser: docling_vision   → Uses vision provider              │
│    parser: glm_ocr          → No VLM (GLM-OCR for scans)        │
│                                                                 │
│  Optional endpoint-backed roles:                                 │
│    synthesizer: null       → no generated answer                 │
│    synthesizer: endpoint/X → fitz answer can synthesize          │
│    query_intelligence: null       → deterministic query prep     │
│    query_intelligence: endpoint/X → optional rewrite/analyze bus │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## VLM (vision-language model) control

The VLM is used by the `docling_vision` parser to describe figures /
images in PDFs during ingestion.

### How it works

1. Set a vision provider in `.fitz/config.yaml`.
2. Choose the parser:
   - `parser: cpu`             → fast default, no VLM
   - `parser: docling`         → figures replaced by `[Figure]`
   - `parser: docling_vision`  → figures get VLM-generated descriptions
   - `parser: glm_ocr`         → no VLM, GLM-OCR for scans

### Config example

```yaml
parser: docling_vision           # parser choice toggles VLM use

vision: endpoint/gpt-4o          # any OpenAI-compatible vision model
vision_base_url: https://api.openai.com/v1
vision_api_key_env: OPENAI_API_KEY
```

### Key files

| File                                              | Purpose                          |
| ------------------------------------------------- | -------------------------------- |
| `fitz_sage/ingestion/parser/router.py`            | Routes by file ext + config      |
| `fitz_sage/ingestion/parser/plugins/docling.py`   | Standard parser (no VLM)         |
| `fitz_sage/ingestion/parser/plugins/docling_vision.py` | VLM-enabled parser          |
| `fitz_sage/ingestion/parser/plugins/glm_ocr.py`   | pdfplumber + GLM-OCR hybrid      |

---

## ONNX reranker control

The reranker is part of the standard retrieval pipeline. It is an INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default) run on raw
`onnxruntime`. It uses two concurrent batch-one passes over a bounded
candidate prefix. No external API call.

### How it works

1. `rerank: onnx` is the default.
2. The retrieval pipeline includes the reranker step before Pyrrho governance.
3. The engine config does not expose a normal "rerank off" mode.
4. `rerank_candidates` controls neural work without shrinking BM25 recall.

### Config example

```yaml
# Default — gte-reranker-modernbert-base
rerank: onnx

# Compatible custom repository with onnx/model_int8.onnx
# rerank: onnx/owner/compatible-reranker
```

### Key files

| File                                              | Purpose                            |
| ------------------------------------------------- | ---------------------------------- |
| `fitz_sage/engines/fitz_krag/retrieval/reranker.py` | Pipeline step                    |
| `fitz_sage/llm/providers/onnx_reranker.py`        | ONNX cross-encoder provider        |
| `fitz_sage/llm/config.py`                         | Provider-spec → instance factory   |

---

## Governance

Epistemic governance (SUFFICIENT / DISPUTED / INSUFFICIENT in Pyrrho v2)
follows the same declaration pattern — the `governance:` key declares the
classifier:

```yaml
governance: pyrrho
# Custom remote models require pyrrho/<owner/repo@40-character-commit>.
# Local model directories use pyrrho/<absolute-path>.
```

The bare value uses Pyrrho's accepted immutable default. Pyrrho owns v2
query-planning heads and the one authoritative decision over the fixed delivered
evidence set; Fitz/KRAG owns the retrieval mechanics that consume planning
signals.

## Managed Qwen

Managed Qwen does not follow endpoint-provider presence. It supplies:

- standard query-time semantic keywords;
- optional background entity and temporal metadata;
- optional hierarchy and demand summaries.

Exact model/runtime details live in [Managed Models](MANAGED_MODELS.md).

There is no `enrichment:` provider key. Source indexing does not load Qwen. A
query-expansion failure is recorded and falls back to the literal plan; a
background failure is reported through enrichment status without weakening the
stored source index.

---

## Why this pattern?

1. **No boolean flags to sync.** Provider presence is the toggle.
2. **Reading the config tells you endpoint usage.** No hidden network calls.
3. **One retrieval pipeline.** Steps are conditionally executed; you
   don't have to swap pipeline plugins to add/remove features.

---

## Maintenance Rule

Every documented key must exist in `FitzKragConfig`. Parser behavior is
selected by `parser`; optional endpoint-backed behavior is controlled by the
existing role keys (`query_intelligence`, `synthesizer`, `vision`, and the
three chat tiers). Do not add illustrative or dormant config keys.

---

## Quick Reference

| Feature | Config key | Product default |
|---------|------------|-----------------|
| Managed Qwen semantic terms | internal | standard local CPU query step |
| Background entity/hierarchy work | internal | optional after source indexing |
| Pyrrho governance | `governance:` | accepted immutable default |
| ONNX reranker | `rerank:` | `rerank: onnx` |
| Answer synthesis | `synthesizer:` | `null`, enabled only by explicit provider |
| Query intelligence | `query_intelligence:` | `null`, deterministic prep + Qwen keywords |
| VLM in parser | `parser:` + `vision:` | off unless `parser: docling_vision` + `vision:` |

---

## See Also

- [Reranking](features/retrieval/reranking.md) — detailed reranker docs
- [Enrichment](ENRICHMENT.md) — managed Qwen enrichment
- [Retrieval Pipeline](RETRIEVAL_PIPELINE.md) — how retrieval stages fit together
- [PLUGINS.md](PLUGINS.md) — supported extension points
- [CONFIG.md](CONFIG.md) — full configuration reference
