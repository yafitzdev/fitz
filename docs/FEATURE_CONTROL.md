# Feature Control Architecture

How optional features (VLM in the parser, LLM reranking) are switched
on and off in fitz-sage **v0.12.0+**.

---

## Design Philosophy

fitz-sage uses a **provider-presence pattern**:

- **Config declares WHICH** provider/model to use.
- **Provider presence determines IF** the feature runs.
- **No `enabled: true / false` flags.** Setting a provider enables
  the feature; omitting it (or setting `null`) skips that step.

This keeps the config declarative and avoids boolean flags that
can drift out of sync with the actual provider config.

The single exception is `enable_guardrails: bool` in
`FitzKragConfig` — used by the smoke test to bypass the constraints
cascade and measure raw retrieval timing.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  CONFIG (~/.fitz/config/<engine>.yaml)                          │
│  Declares WHICH provider/model to use                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  vision: endpoint                rerank: onnx   │
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
│    parser: docling          → No VLM (figures become "[Figure]")│
│    parser: docling_vision   → Uses vision provider              │
│    parser: glm_ocr          → No VLM (fast, default)            │
│                                                                 │
│  ONNX Reranker (controlled by `rerank:` presence):               │
│    rerank: (omitted / null) → No reranker step                  │
│    rerank: onnx → Reranker auto-enabled         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## VLM (vision-language model) control

The VLM is used by the `docling_vision` parser to describe figures /
images in PDFs during ingestion.

### How it works

1. Set a vision provider in `~/.fitz/config/fitz_krag.yaml`.
2. Choose the parser:
   - `parser: docling`         → figures replaced by `[Figure]`
   - `parser: docling_vision`  → figures get VLM-generated descriptions
   - `parser: glm_ocr`         → fast default; no VLM, GLM-OCR for scans

### Config example

```yaml
parser: docling_vision           # parser choice toggles VLM use

vision: endpoint                 # any OpenAI-compatible vision model
vision_base_url: https://api.openai.com/v1
vision_api_key_env: OPENAI_API_KEY
vision_model: gpt-4o
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

The reranker is an INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default) served via
`optimum.onnxruntime`. One forward pass over `(query, candidate)`
pairs; ~30–100 ms on CPU for 10–20 candidates. No external API call.

### How it works

1. Set `rerank:` in the engine config (default: `onnx`).
2. The retrieval pipeline auto-includes the reranker step when the
   provider is configured; otherwise it's skipped.

### Config example

```yaml
# Default — gte-reranker-modernbert-base
rerank: onnx

# Different cross-encoder
# rerank: onnx/BAAI/bge-reranker-base
# rerank: onnx/jinaai/jina-reranker-v3

# Disabled
# rerank: null    # or omit the key entirely
```

### Key files

| File                                              | Purpose                            |
| ------------------------------------------------- | ---------------------------------- |
| `fitz_sage/engines/fitz_krag/retrieval/reranker.py` | Pipeline step                    |
| `fitz_sage/llm/providers/onnx_reranker.py`        | ONNX cross-encoder provider        |
| `fitz_sage/llm/config.py`                         | Provider-spec → instance factory   |

---

## Constraints

The epistemic constraint cascade (TRUSTWORTHY / DISPUTED / ABSTAIN) is
controlled by a **list of constraint plugins**, not provider presence:

```yaml
constraints:
  - conflict_aware
  - insufficient_evidence
  - causal_attribution
  - specific_info_type
```

Constraints are presence-controlled by entry in the list. To run
without constraints, set `enable_guardrails: false` (the boolean
escape hatch used by the smoke test).

---

## Why this pattern?

1. **No boolean flags to sync.** Provider presence is the toggle.
2. **Reading the config tells you the runtime.** No hidden defaults
   silently flipping behaviour.
3. **One retrieval pipeline.** Steps are conditionally executed; you
   don't have to swap pipeline plugins to add/remove features.

---

## Adding a new optional feature

Pattern:

1. **Ingestion-time** (like VLM): create two parser plugins, let
   `parser:` pick.
2. **Query-time**: add a config dependency (e.g. `summarizer:`) and
   skip the pipeline step when the dependency is absent.

Sketch for a hypothetical query-time summarizer:

```python
# In the pipeline step
if config.summarizer is None:
    return inputs   # passthrough — feature disabled
summarizer = build_summarizer(config.summarizer)
return summarizer.run(inputs)
```

```yaml
# Config to switch it on
summarizer: endpoint/summarizer
```

---

## Quick Reference

| Feature        | Config key   | Enable                          | Disable                       |
| -------------- | ------------ | ------------------------------- | ----------------------------- |
| VLM in parser  | `parser:` + `vision:` | `parser: docling_vision` + `vision:` set | `parser: docling` / `parser: glm_ocr` |
| ONNX reranker  | `rerank:`    | `rerank: onnx` (default)        | `rerank: null` (or omit)      |
| Guardrails     | `enable_guardrails`   | `true` (default)       | `false` (smoke test only)     |

---

## See Also

- [Reranking](features/retrieval/reranking.md) — detailed reranker docs
- [PLUGINS.md](PLUGINS.md) — plugin development guide
- [CONFIG.md](CONFIG.md) — full configuration reference
