<!-- docs/CLI.md -->
# CLI Reference

fitz-sage **v0.14.1+**. The CLI is one binary, `fitz`, with a small
set of commands.

```bash
fitz --help
fitz <command> --help
```

---

## Quick Start

```bash
# Start the required local enrichment runtime once before source-backed retrieval
llama-server -hf bartowski/Qwen_Qwen3.5-0.8B-GGUF:Q4_K_M \
  --alias qwen3.5-0.8b@Q4_K_M \
  --host 127.0.0.1 --port 8080

# One-shot: register docs + retrieve governed evidence
fitz retrieve "What is X?" --source ./docs

# Subsequent retrieval reuses the collection
fitz retrieve "Follow-up question"

# JSON emits the full EvidencePack
fitz retrieve "What is X?" --source ./docs --format json

# Optional synthesis: pass a provider spec directly
fitz answer "What is X?" --synthesizer openai/gpt-4o --source ./docs

# Or point at any OpenAI-compatible endpoint
fitz answer "What is X?" \
  --endpoint https://api.together.xyz/v1 \
  --synthesizer endpoint/meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY \
  --source ./docs
```

---

## Commands

The main commands are `retrieve`, `answer`, `query`, `collections`, and `serve`.
Configuration is auto-created on first run; there is no separate init
step.

### `fitz retrieve`

Return a ranked, governed evidence pack without answer synthesis. With
`--source`, registers documents first; without it, retrieves from the active
collection.

```bash
fitz retrieve "Your question"
fitz retrieve "What is this about?" --source ./docs
fitz retrieve "Which test failed?" -c my_collection --source ./docs --top-k 10
fitz retrieve "Which test failed?" --format json
```

**Arguments**
- `QUESTION` — the question to retrieve evidence for

**Options**
- `-s, --source PATH` — register documents (file or directory) before retrieval
- `-c, --collection TEXT` — collection name (default: `default`)
- `-e, --engine TEXT` — engine name (default: `fitz_krag`)
- `--format text|json` — human-readable output or serialized `EvidencePack`
- `--top-k INT` — maximum evidence items to show

### `fitz answer`

Generate a synthesized answer from the retrieved evidence. This requires an
explicit synthesizer provider, either from config (`synthesizer:`) or from the
CLI synthesis flags.

```bash
fitz answer "Your question" --source ./docs \
  --synthesizer openai/gpt-4o

fitz answer "Your question" --source ./docs \
  --endpoint http://localhost:8080/v1 \
  --synthesizer endpoint/qwen3.5-0.8b@Q4_K_M

fitz answer "Your question" -c my_collection \
  --synthesizer openai/gpt-4o
```

If no synthesizer is configured, the command fails with an actionable error and
points you back to `fitz retrieve`.

**Options**
- `-s, --source PATH` — register documents before answering
- `-c, --collection TEXT` — collection name
- `-e, --engine TEXT` — engine name
- `--synthesizer TEXT` — provider/model spec for synthesis
- `--endpoint TEXT` — OpenAI-compatible URL; pairs with `--model` or `--synthesizer endpoint/<model>`
- `-m, --model TEXT` — chat model name sent to the endpoint
- `--api-key-env TEXT` — env var holding the API key

### `fitz query`

Compatibility alias for synthesized answer behavior, plus interactive chat mode.
For new workflows, prefer `fitz retrieve` for evidence and `fitz answer` for
explicit synthesis.

```bash
fitz query "Your question" --synthesizer openai/gpt-4o
fitz query "Your question" --endpoint http://localhost:8080/v1 --synthesizer endpoint/qwen3.5-0.8b@Q4_K_M
fitz query --chat -c my_collection
```

---

### `fitz collections`

Manage collections (list, info, delete).

```bash
fitz collections          # interactive menu
fitz collections list
fitz collections info my_collection
fitz collections delete my_collection
```

A collection is a single `.db` file under `~/.fitz/sqlite/fitz_<name>.db`.
`delete` removes the file (and its `-wal` / `-shm` siblings) — there's no
DB-level `DROP DATABASE` step because there's no server.

---

### `fitz serve`

Start the REST API server.

```bash
fitz serve
fitz serve --host 0.0.0.0 -p 8080
fitz serve --reload                  # auto-reload (dev)
```

**Options**
- `-h, --host TEXT` — bind host (default `127.0.0.1`)
- `-p, --port INT` — bind port (default `8000`)
- `--reload` — auto-reload on code change

**Endpoints** (see [API.md](API.md) for the full schema)
- `POST /query` — query the knowledge base; pass `source` to register first
- `POST /chat` — multi-turn conversation
- `GET /collections` — list
- `GET /collections/{name}` — details
- `DELETE /collections/{name}` — delete
- `GET /health` — health check

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
enricher: endpoint/qwen3.5-0.8b@Q4_K_M
summarizer: endpoint/qwen3.5-0.8b@Q4_K_M
```

This file is auto-created on first run. See [CONFIG.md](CONFIG.md) for
every key and [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) for
ready-to-paste configurations.

---

## Environment Variables

API keys are read from environment variables (never put them in
config). The CLI looks up the variable name from `chat_api_key_env`,
or from the `--api-key-env` flag.

```bash
# OpenAI
export OPENAI_API_KEY="..."

# Together / Groq / Mistral
export TOGETHER_API_KEY="..."

# Local servers (no key)
# llama-server --port 8080
# ollama serve
# LM Studio (Settings → Developer → Local Server)
```

`FITZ_HOME` overrides `~/.fitz/` if you want to relocate config +
storage.

`FITZ_LOG_LEVEL=DEBUG` enables verbose logging for any command.

---

## Common Workflows

### Local-first setup

```bash
# Ingest + retrieve governed evidence with the required local Qwen server running
fitz retrieve "What's in my docs?" --source ./docs
```

### Multi-turn exploration

```bash
fitz query --chat --source ./docs -c project_x
# ... or pick up an existing collection ...
fitz query --chat -c project_x
```

### Cloud-only

```bash
export OPENAI_API_KEY=...
fitz answer "What is X?" \
  --synthesizer openai/gpt-4o \
  --source ./docs
```

---

## Getting Help

```bash
fitz --help
fitz <command> --help
```

See also: [CONFIG.md](CONFIG.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md),
[API.md](API.md).
