<!-- docs/MANAGED_MODELS.md -->
# Managed Models

fitz-sage retrieval works without an API key or an external inference server.
The standard retrieval path uses local CPU models.

## Local Models

| Job | Model | Runtime | Why it exists |
|---|---|---|---|
| Semantic query terms and background enrichment | `onnx-community/Qwen3-0.6B-DQ-ONNX` (`qwen3-0.6b`) | `onnxruntime-genai`, CPU | Standard query expansion plus optional entity/hierarchy metadata. |
| Reranking | `Alibaba-NLP/gte-reranker-modernbert-base` | raw `onnxruntime`, CPU | Cross-encoder precision over broad recall candidates. |
| Governance | `yafitzdev/pyrrho-v2-nano-g1` at revision `948f0500b74871cfaec7689a01d4eab0dd516e1b` | raw `onnxruntime`, CPU | Accepted immutable Pyrrho default; custom local or commit-pinned models are supported. |

None of these models require `optimum`, `llama.cpp`, GGUF, or an
OpenAI-compatible server. Qwen uses ONNX Runtime GenAI; the reranker and Pyrrho
v2 load pre-built ONNX graphs through plain ONNX Runtime.

## Download Behavior

Qwen, the reranker, and Pyrrho are downloaded lazily into the Hugging Face
cache.

| Model | Trigger |
|---|---|
| Qwen3 0.6B ONNX GenAI | First background-enrichment or semantic-query-keyword operation. `point()` does not load it. |
| ONNX reranker | First retrieval pass that has enough candidates to rerank. |
| Pyrrho v2 | First query-plan or evidence-decision call. |

After the first download, subsequent runs reuse the cached snapshots.

## Local Smoke Check

Run the standard local CPU path against a tiny generated corpus:

```bash
python tools/smoke_local_retrieval.py
```

The script resolves managed Qwen, indexes the corpus, runs background
enrichment, and executes queries through the reranker and Pyrrho. It is a
runtime smoke check, not a retrieval-quality benchmark.

## Offline and Air-Gapped Use

For disconnected deployments, warm the managed models on a connected machine,
then copy the caches to the target machine before running queries.

Warm the standard model set:

```bash
python -c "from fitz_sage.llm.providers.onnx_chat import OnnxChat; OnnxChat().ensure_available(include_checksum=True)"
python -c "from fitz_sage.llm.providers.onnx_reranker import OnnxReranker; OnnxReranker().rerank('warmup', ['one', 'two'])"
python -c "from fitz_sage.integrations.pyrrho import create_pyrrho; create_pyrrho('pyrrho').decide('warmup', [{'source_id': 'warmup', 'text': 'warmup evidence'}])"
```

Copy the model cache:

| Cache | What it contains |
|---|---|
| Hugging Face cache (`HF_HOME`, or the platform default) | Qwen and Pyrrho snapshots, reranker ONNX file, tokenizers, and configs |

On the target machine, point `HF_HOME` at the copied Hugging Face cache and set
Hugging Face offline mode:

```bash
export HF_HOME=/opt/fitz/hf-cache
export HF_HUB_OFFLINE=1
```

Alternatively, copy an unpacked custom Pyrrho model and configure its exact
local path:

```yaml
governance: pyrrho//opt/fitz/models/pyrrho/custom-release
```

The double slash after `pyrrho/` is intentional for absolute Unix paths. On
Windows, use a normal absolute path after the provider prefix, for example
`pyrrho/C:\fitz\models\pyrrho\custom-release`.

## Qwen Responsibilities

Qwen enrichment is not a user-selected endpoint provider. It is part of the
local retrieval product:

- query-time semantic keyword expansion
- optional entity and temporal metadata
- optional L1 file summaries
- optional L2 corpus hierarchy
- optional demand summaries for surfaced files

`point()` never waits for Qwen. If query expansion fails, Fitz-Sage records the
failure and continues with the literal prepared plan. Background failures are
reported in `indexing_status().enrichment` while the source index remains
searchable.

## Optional Synthesis Is Separate

Answer synthesis is not part of `fitz retrieve`. If a user explicitly runs
`fitz answer`, synthesis may use an OpenAI-compatible endpoint configured by the
user. That endpoint is separate from the managed ONNX models above.

## Inspecting Managed Qwen

The managed Qwen provider exposes model metadata for smoke checks:

```python
from fitz_sage.llm.providers.onnx_chat import OnnxChat

info = OnnxChat().model_info(include_checksum=True)
print(info.repo_id)
print(info.revision)
print(info.onnx_path)
print(info.bundle_sha256)
```
