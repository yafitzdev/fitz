<!-- docs/MANAGED_MODELS.md -->
# Managed Models

fitz-sage retrieval works without an API key or an external inference server.
The standard retrieval path uses local CPU models.

## Default Local Models

| Job | Model | Runtime | Why it exists |
|---|---|---|---|
| Enrichment and semantic query keywords | `onnx-community/Qwen3.5-0.8B-Text-ONNX` (`qwen3.5-0.8b`) | raw `onnxruntime`, CPU | Required metadata backbone for better recall. |
| Reranking | `Alibaba-NLP/gte-reranker-modernbert-base` | raw `onnxruntime`, CPU | Cross-encoder precision over broad recall candidates. |
| Governance + query signals | `yafitzdev/pyrrho-nano-g3.1` | `torch` + `safetensors`, CPU | Sufficiency/conflict classifier for ranked evidence prefixes plus pre-retrieval query metadata. |

None of these models require `optimum`, `llama.cpp`, GGUF, or an
OpenAI-compatible server. Pyrrho g3.1 currently uses `torch` because the
published multitask checkpoint is a safetensors model; Qwen and the reranker
remain ONNX-only.

## Download Behavior

Models are downloaded lazily from Hugging Face the first time a workflow needs
them. Qwen and the reranker use the Hugging Face cache. Pyrrho uses Fitz's
managed user cache under `~/.fitz/models/pyrrho/...` so Windows users do not need
symlink privileges.

| Model | Trigger |
|---|---|
| Qwen3.5 0.8B ONNX | First query or ingest that needs required enrichment or semantic query keywords. |
| ONNX reranker | First retrieval pass that has enough candidates to rerank. |
| Pyrrho g3.1 | First query-signal classification or governance cutoff evaluation. |

The CLI may print messages such as:

```text
Preparing managed Qwen3.5 0.8B ONNX enrichment model...
Managed Qwen ready (<revision>).
```

After the first download, subsequent runs reuse the cached snapshots.

## Qwen Is Mandatory

Qwen enrichment is not a user-selected optional provider. It is part of the
retrieval product:

- query-time semantic keyword expansion
- file keyword and alias extraction
- entity linking
- L1 file summaries
- L2 corpus hierarchy
- demand summaries for surfaced files

The foreground query path can return before all deep enrichment is complete, but
the background daemon must continue the required enrichment work.

## Optional Synthesis Is Separate

Answer synthesis is not part of `fitz query`. If a user explicitly runs
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
