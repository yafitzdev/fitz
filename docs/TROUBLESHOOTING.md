# Troubleshooting

Fitz-Sage keeps source indexing, optional background enrichment, local query
models, and optional endpoint roles separate. Start by identifying which
boundary failed.

## Quick Checks

1. Inspect `.fitz/config.yaml` in the directory where the command runs.
2. Run with `FITZ_LOG_LEVEL=DEBUG`.
3. Inspect collection status through the SDK or REST API.
4. Distinguish `query_ready` from `enrichment.complete`.

```python
from fitz_sage import fitz

client = fitz(collection="docs")
client.point("./docs")
print(client.indexing_status())
```

`point()` returns after supported files are indexed or explicitly failed. It
does not wait for Qwen entity or hierarchy enrichment.

## Endpoint Connection Errors

`fitz retrieve` does not need an external chat server. An endpoint connection
matters only when the config enables a role such as:

```yaml
query_intelligence: endpoint/qwen2.5-7b-instruct
synthesizer: endpoint/qwen2.5-7b-instruct
vision: endpoint/gpt-4o
```

Check the configured URL and model:

```bash
curl http://localhost:8080/v1/models
```

For Ollama's OpenAI-compatible chat protocol, use a `/v1` URL such as
`http://localhost:11434/v1`. The `glm_ocr` parser is different: its scan
fallback talks to Ollama's native API at `http://localhost:11434` and expects
the `glm-ocr` model.

## Missing API Key

Hosted keys are read from environment variables. The config stores the variable
name, not the secret:

```yaml
chat_api_key_env: OPENAI_API_KEY
```

```bash
# Linux/macOS
export OPENAI_API_KEY="..."

# PowerShell
$env:OPENAI_API_KEY = "..."
```

Leave the key field unset for an unauthenticated local endpoint.

## Managed Model Download Or Load Failure

The managed components load lazily:

| Component | First trigger | Failure behavior |
|---|---|---|
| Qwen | semantic query terms or background enrichment | query expansion falls back to literal plan; background failure is reported |
| ONNX reranker | a retrieval pool large enough to rerank | query fails because reranking is part of the product path |
| Pyrrho | PRE planning or final evidence decision | query fails because governance is mandatory |

Check network/proxy access to Hugging Face during the first download, available
disk space, and the configured Hugging Face cache (`HF_HOME`). For an
air-gapped deployment, warm and copy the caches as described in
[Managed Models](MANAGED_MODELS.md).

## Source Files Are Missing Or Unsearchable

Inspect `indexing_status()` instead of assuming discovery succeeded:

- `unsupported_files` lists extensions outside the active parser/config contract;
- `failed_files` lists supported files that failed parsing or storage;
- `indexed` counts searchable supported files;
- `query_ready` means no supported file remains pending.

The default CPU parser supports embedded text in PDF, DOCX, and PPTX. Common
boundaries:

- image-only PDF or PPTX files need OCR/vision selection;
- `.xlsx` is disabled under `parser: cpu` and is not a native table format;
- native CSV/TSV expects a usable first-row header;
- empty source files are reported instead of indexed as empty evidence;
- code-language settings can exclude otherwise recognized code extensions.

See [Ingestion](INGESTION.md) for the authoritative format matrix.

## Source Index Is Ready But Enrichment Is Pending

This is normal. Evidence retrieval uses the persisted sections, symbols, and
tables immediately. `indexing_status()["enrichment"]` reports optional entity,
hierarchy, and demand-summary progress.

For a long-lived Python process:

```python
from fitz_sage import fitz

client = fitz(collection="my_docs")
client.wait_for_enrichment()
```

The CLI can hand pending work to its hidden enrichment daemon after returning
evidence. An enrichment failure does not remove source-index data.

## SQLite Path Or Lock Errors

The default workspace is `<current-directory>/.fitz/`, not a global home
directory. Confirm the process can create and write:

```text
.fitz/sqlite/fitz_<collection>.db
.fitz/collections/<collection>/manifest.json
```

SQLite uses WAL mode and a 30-second busy timeout. Multiple readers are allowed,
but writes to one collection are serialized. Do not manually delete only the
main `.db` while a process is using its `-wal`/`-shm` sidecars.

Use the interactive collection manager to delete a collection cleanly:

```bash
fitz collections
```

The REST API also exposes `DELETE /collections/{name}`.

## Old Collection Schema

Fitz-Sage has no compatibility migration promise for unreleased/old collection
schemas. Delete the collection through the collection manager and point the
source again. Source files remain the authority.

## Slow Queries

Use a retrieval trace before changing budgets:

```bash
fitz retrieve "question" --source ./docs --trace run.json
fitz explain run.json
```

Common costs are managed Qwen query expansion, cross-encoder reranking, Pyrrho,
and repeated evidence closure. On very large section indexes, closure recall can
dominate. Lowering `rerank_candidates`, `top_addresses`, or `top_read` trades
coverage for latency; compare evidence quality before keeping that change.

Cold queries also include lazy model load and may include model download. See
[Benchmarks](BENCHMARK.md) for current warm measurements.

## Answer Command Has No Synthesizer

`fitz retrieve` returns evidence without generation. `fitz answer` requires a
configured synthesizer or invocation flags:

```bash
fitz answer "What is X?" \
  --source ./docs \
  --synthesizer openai/gpt-4o
```

This requirement is deliberate; the retrieval package does not silently choose
a prose-generation endpoint.

## Debugging

Enable logs:

```bash
# Linux/macOS
FITZ_LOG_LEVEL=DEBUG fitz retrieve "test" --source ./docs

# PowerShell
$env:FITZ_LOG_LEVEL = "DEBUG"
fitz retrieve "test" --source ./docs
```

Inspect the manifest:

```bash
python -m json.tool .fitz/collections/<collection>/manifest.json
```

Test an endpoint directly:

```python
from fitz_sage.llm.client import get_chat

chat = get_chat(
    "endpoint/qwen2.5-7b-instruct",
    config={"base_url": "http://localhost:8080/v1"},
)
print(chat.chat([{"role": "user", "content": "Hello"}]))
```

Inspect current SQLite tables:

```bash
sqlite3 .fitz/sqlite/fitz_<collection>.db
.tables
SELECT COUNT(*) FROM krag_section_index;
SELECT id, title FROM krag_section_index LIMIT 5;
```

## Reporting A Problem

Include:

- `pip show fitz-sage`;
- `python --version` and operating system;
- the full traceback;
- redacted `.fitz/config.yaml`;
- source-index and enrichment status;
- a redacted retrieval trace when the problem is ranking or governance.

Report issues at [GitHub Issues](https://github.com/yafitzdev/fitz-sage/issues).

## Related

- [Configuration](CONFIG.md)
- [CLI](CLI.md)
- [Ingestion](INGESTION.md)
- [Retrieval Runs](RETRIEVAL_RUNS.md)
- [Limitations](LIMITATIONS.md)
