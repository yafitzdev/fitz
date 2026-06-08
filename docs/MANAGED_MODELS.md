<!-- docs/MANAGED_MODELS.md -->
# Managed Models

fitz-sage retrieval works without an API key or an external inference server.
The standard retrieval path uses local CPU models.

## Default Local Models

| Job | Model | Runtime | Why it exists |
|---|---|---|---|
| Enrichment and semantic query keywords | `onnx-community/Qwen3.5-0.8B-Text-ONNX` (`qwen3.5-0.8b`) | raw `onnxruntime`, CPU | Required metadata backbone for better recall. |
| Reranking | `Alibaba-NLP/gte-reranker-modernbert-base` | raw `onnxruntime`, CPU | Cross-encoder precision over broad recall candidates. |
| Governance + query signals | `yafitzdev/pyrrho-nano-g4-alpha` | `torch` + `safetensors`, CPU | Sufficiency/conflict classifier for ranked evidence prefixes plus pre-retrieval query and evidence signal metadata. |

None of these models require `optimum`, `llama.cpp`, GGUF, or an
OpenAI-compatible server. Pyrrho g4-alpha uses `torch` because the
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
| Pyrrho g4-alpha | First query-signal classification or governance cutoff evaluation. |

The CLI may print messages such as:

```text
Preparing managed Qwen3.5 0.8B ONNX enrichment model...
Managed Qwen ready (<revision>).
```

After the first download, subsequent runs reuse the cached snapshots.

## Offline and Air-Gapped Use

For disconnected deployments, warm the managed models on a connected machine,
then copy the caches to the target machine before running queries.

Warm the standard model set:

```bash
python -c "from fitz_sage.llm.providers.onnx_chat import OnnxChat; OnnxChat().ensure_available(include_checksum=True)"
python -c "from fitz_sage.llm.providers.onnx_reranker import OnnxReranker; OnnxReranker().rerank('warmup', ['one', 'two'])"
python -c "from fitz_sage.governance import create_governance; create_governance('pyrrho').classify_query('warmup')"
```

Copy both cache roots:

| Cache | What it contains |
|---|---|
| Hugging Face cache (`HF_HOME`, or the platform default) | Qwen ONNX snapshot, reranker ONNX file, tokenizers, configs |
| `~/.fitz/models/pyrrho/` | Managed Pyrrho safetensors package |

On the target machine, point `HF_HOME` at the copied Hugging Face cache and set
Hugging Face offline mode:

```bash
export HF_HOME=/opt/fitz/hf-cache
export HF_HUB_OFFLINE=1
```

For Pyrrho, either copy `~/.fitz/models/pyrrho/` to the same Fitz home location
or configure governance with the unpacked local package path:

```yaml
governance: pyrrho//opt/fitz/models/pyrrho/yafitzdev__pyrrho-nano-g4-alpha
```

The double slash after `pyrrho/` is intentional for absolute Unix paths. On
Windows, use a normal absolute path after the provider prefix, for example
`pyrrho/C:\fitz\models\pyrrho\yafitzdev__pyrrho-nano-g4-alpha`.

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
