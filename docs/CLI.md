<!-- docs/CLI.md -->
# CLI Reference

The CLI is intentionally small: one command for the normal retrieval workflow,
one command for explicit answer synthesis, and a few operational commands.

```bash
fitz --help
fitz <command> --help
```

---

## Quick Start

```bash
# From inside a document folder: register the current directory and return evidence
fitz query "Which documents are relevant to the refund policy?"

# Point at a different source folder
fitz query "Which documents are relevant to the refund policy?" --source ./docs

# Advanced evidence controls
fitz retrieve "Which documents are relevant?" --source ./docs --top-k 8
fitz retrieve "Which documents are relevant?" --source ./docs --format json

# Optional generated answer, only when you explicitly provide a synthesizer
fitz answer "What is the refund policy?" --source ./docs \
  --endpoint http://localhost:8080/v1 \
  --synthesizer endpoint/qwen2.5-7b-instruct
```

`fitz query` is the product default. It returns a ranked `EvidencePack`; it does
not generate an answer. Required retrieval enrichment runs through the managed
Qwen3 0.6B ONNX GenAI model on CPU, so no API key or external inference server is
needed for retrieval.

---

## Commands

The main commands are `query`, `retrieve`, `answer`, `collections`, and `serve`.
Configuration is auto-created on first run; there is no separate init or ingest
step.

### `fitz query`

Return governed evidence with the fewest flags. If neither `--source` nor
`--collection` is provided, `fitz query` uses the current working directory as
the source and derives the collection name from that folder.

```bash
fitz query "Your question"
fitz query "Your question" --source ./docs
fitz query "Your question" --collection product_docs
fitz query "Your question" --source ./docs --collection product_docs
```

**Arguments**
- `QUESTION` - the question to retrieve evidence for

**Options**
- `-s, --source PATH` - file or directory to register before retrieval
- `-c, --collection TEXT` - collection name; defaults to the source folder name
- `-e, --engine TEXT` - engine name; defaults to `fitz_krag`

**What the user sees**

1. A short progress feed: source registration, managed Qwen readiness, parsing,
   query analysis, and retrieval.
2. A ranked evidence table.
3. Pyrrho governance metadata folded into the table caption: probabilities,
   cutoff policy, and reasons.
4. An indexing status line when Qwen keyword/deep enrichment is still running.

If enrichment is still pending after the first evidence pack is shown, the CLI
starts a detached `index-daemon` process so the collection keeps improving after
the foreground command exits.

### `fitz retrieve`

Same evidence workflow as `fitz query`, with explicit output controls. Use it
when scripts need JSON or a fixed evidence count.

```bash
fitz retrieve "Your question"
fitz retrieve "What is this about?" --source ./docs
fitz retrieve "Which test failed?" -c my_collection --source ./docs --top-k 10
fitz retrieve "Which test failed?" --format json
```

**Options**
- `-s, --source PATH` - file or directory to register before retrieval
- `-c, --collection TEXT` - collection name; defaults to the source folder name
- `-e, --engine TEXT` - engine name; defaults to `fitz_krag`
- `--format text|json` - human-readable table or serialized `EvidencePack`
- `--top-k INT` - maximum evidence items to show

### `fitz answer`

Generate an optional synthesized answer from retrieved evidence. This is not the
default retrieval surface. It requires a configured `synthesizer:` provider or
CLI synthesis flags.

```bash
fitz answer "Your question" --source ./docs \
  --synthesizer openai/gpt-4o

fitz answer "Your question" --source ./docs \
  --endpoint http://localhost:8080/v1 \
  --synthesizer endpoint/qwen2.5-7b-instruct

fitz answer "Your question" -c my_collection \
  --synthesizer openai/gpt-4o
```

If no synthesizer is configured, the command fails with an actionable error and
points you back to evidence retrieval.

**Options**
- `-s, --source PATH` - register documents before answering
- `-c, --collection TEXT` - collection name
- `-e, --engine TEXT` - engine name
- `--synthesizer TEXT` - provider/model spec for synthesis
- `--endpoint TEXT` - OpenAI-compatible URL; pairs with `--model` or
  `--synthesizer endpoint/<model>`
- `-m, --model TEXT` - chat model name sent to the endpoint
- `--api-key-env TEXT` - env var holding the API key

### `fitz collections`

Manage collections.

```bash
fitz collections          # interactive menu
fitz collections list
fitz collections info my_collection
fitz collections delete my_collection
```

A collection is a single SQLite database under the fitz workspace. Deleting a
collection removes the `.db` file and its `-wal` / `-shm` siblings.

### `fitz serve`

Start the REST API server.

```bash
fitz serve
fitz serve --host 0.0.0.0 -p 8080
fitz serve --reload
```

**Options**
- `-h, --host TEXT` - bind host; default `127.0.0.1`
- `-p, --port INT` - bind port; default `8000`
- `--reload` - auto-reload on code change

See [API.md](API.md) for endpoint schemas.

---

## Configuration

The minimum on-disk config (`~/.fitz/config/fitz_krag.yaml`) is:

```yaml
collection: default
parser: cpu
rerank: onnx
governance: pyrrho
query_intelligence: null
synthesizer: null
chat_base_url: http://127.0.0.1:8080/v1
```

This is enough for `fitz query`, `fitz retrieve`, and
`fitz_sage.evidence(...)`. Managed Qwen3 0.6B ONNX GenAI enrichment, the ONNX
reranker, and Pyrrho governance all run locally on CPU.

See [CONFIG.md](CONFIG.md) for every key and
[CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) for deployment examples.

---

## Environment Variables

API keys are needed only for optional endpoint-backed roles such as answer
synthesis, query intelligence, or vision parsing. The CLI looks up the variable
name from `chat_api_key_env` or from the `--api-key-env` flag.

```bash
# OpenAI
export OPENAI_API_KEY="..."

# Together / Groq / Mistral
export TOGETHER_API_KEY="..."
```

`FITZ_HOME` overrides `~/.fitz/` if you want to relocate config and storage.
`FITZ_LOG_LEVEL=DEBUG` enables verbose logging for any command.

---

## See Also

- [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) - retrieval flow and indexing states
- [CONFIG.md](CONFIG.md) - configuration reference
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - common issues
- [API.md](API.md) - REST API reference
