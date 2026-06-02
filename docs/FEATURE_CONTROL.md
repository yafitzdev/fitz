# Feature Control Architecture

How provider-backed features (VLM in the parser, optional synthesis, optional
query intelligence, and advanced model swaps) are declared in fitz-sage
**v0.14.1+**.

---

## Design Philosophy

fitz-sage uses a **provider-presence pattern**:

- **Config declares WHICH** provider/model to use.
- **Provider presence determines IF** optional endpoint-backed features run.
- **No `enabled: true / false` flags.** Setting a provider enables
  the feature; omitting it (or setting `null`) skips that step.
- **Retrieval intelligence is baked in.** Managed Qwen enrichment, broad recall,
  ONNX reranking, and Pyrrho governance are the standard product pipeline.

This keeps the config declarative and avoids boolean flags that
can drift out of sync with the actual provider config.

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

1. Set a vision provider in `~/.fitz/config/fitz_krag.yaml`.
2. Choose the parser:
   - `parser: cpu`             → fast default, no VLM
   - `parser: docling`         → figures replaced by `[Figure]`
   - `parser: docling_vision`  → figures get VLM-generated descriptions
   - `parser: glm_ocr`         → no VLM, GLM-OCR for scans

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

The reranker is part of the standard retrieval pipeline. It is an INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default) run on raw
`onnxruntime`. One forward pass over `(query, candidate)` pairs;
~30–100 ms on CPU for 10–20 candidates. No external API call.

### How it works

1. `rerank: onnx` is the default.
2. The retrieval pipeline includes the reranker step before Pyrrho governance.
3. The engine config does not expose a normal "rerank off" mode.

### Config example

```yaml
# Default — gte-reranker-modernbert-base
rerank: onnx

# Different cross-encoder
# rerank: onnx/BAAI/bge-reranker-base
# rerank: onnx/jinaai/jina-reranker-v3
```

### Key files

| File                                              | Purpose                            |
| ------------------------------------------------- | ---------------------------------- |
| `fitz_sage/engines/fitz_krag/retrieval/reranker.py` | Pipeline step                    |
| `fitz_sage/llm/providers/onnx_reranker.py`        | ONNX cross-encoder provider        |
| `fitz_sage/llm/config.py`                         | Provider-spec → instance factory   |

---

## Governance

Epistemic governance (TRUSTWORTHY / DISPUTED / ABSTAIN) follows the
same declaration pattern — the `governance:` key declares the classifier:

```yaml
governance: pyrrho                  # default — the pyrrho INT8 ONNX classifier
# governance: pyrrho/<hf-model-id>  # a custom pyrrho fine-tune
```

`governance: pyrrho` runs INT8 ONNX evidence-prefix checks in the cutoff loop.
The v0.13.0 constraint+sklearn cascade it replaced is gone.

## Managed enrichment

Qwen3.5 0.8B ONNX enrichment does not follow provider presence because it is
not optional. It is the standard local runtime for:

- ingestion keywords and aliases;
- entity extraction for the entity graph;
- hierarchy summaries;
- default semantic query keywords.

There is no `enrichment:` provider key. If the managed runtime cannot load,
ingestion fails closed instead of silently producing an under-enriched index.

---

## Why this pattern?

1. **No boolean flags to sync.** Provider presence is the toggle.
2. **Reading the config tells you endpoint usage.** No hidden network calls.
3. **One retrieval pipeline.** Steps are conditionally executed; you
   don't have to swap pipeline plugins to add/remove features.

---

## Adding a new optional feature

Pattern:

1. **Ingestion-time** (like VLM): create two parser plugins, let
   `parser:` pick.
2. **Query-time**: add a config dependency (e.g. `answer_expander:`) and
   skip the pipeline step when the dependency is absent.

Sketch for a hypothetical query-time answer expander:

```python
# In the pipeline step
if config.answer_expander is None:
    return inputs   # passthrough — feature disabled
expander = build_answer_expander(config.answer_expander)
return expander.run(inputs)
```

```yaml
# Config to switch it on
answer_expander: endpoint/expander
```

---

## Quick Reference

| Feature | Config key | Product default |
|---------|------------|-----------------|
| Managed Qwen enrichment | internal | always required |
| Pyrrho governance | `governance:` | `governance: pyrrho` |
| ONNX reranker | `rerank:` | `rerank: onnx` |
| Answer synthesis | `synthesizer:` | `null`, enabled only by explicit provider |
| Query intelligence | `query_intelligence:` | `null`, deterministic prep + Qwen keywords |
| VLM in parser | `parser:` + `vision:` | off unless `parser: docling_vision` + `vision:` |

---

## See Also

- [Reranking](features/retrieval/reranking.md) — detailed reranker docs
- [Enrichment](ENRICHMENT.md) — managed Qwen enrichment
- [Retrieval Pipeline](RETRIEVAL_PIPELINE.md) — how retrieval stages fit together
- [PLUGINS.md](PLUGINS.md) — plugin development guide
- [CONFIG.md](CONFIG.md) — full configuration reference
