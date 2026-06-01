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
# One-shot: register docs + query
fitz query "What is X?" --source ./docs

# Subsequent queries reuse the collection
fitz query "Follow-up question"

# Point at any OpenAI-compatible endpoint without editing config
fitz query "What is X?" \
  --endpoint https://api.together.xyz/v1 \
  --model meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY \
  --source ./docs
```

---

## Commands

The CLI has three commands: `query`, `collections`, and `serve`.
Configuration is auto-created on first run; there is no separate init
step.

### `fitz query`

Ask a question. With `--source`, registers documents first; without
it, queries the active collection.

```bash
fitz query "Your question"
fitz query "What is this about?" --source ./docs
fitz query "Summarize the key points" -c my_collection --source ./docs
fitz query --chat                                # interactive
fitz query --chat -c my_collection               # interactive on a collection
```

**Arguments**
- `QUESTION` — the question (optional with `--chat`)

**Options**
- `-s, --source PATH` — register documents (file or directory) before querying
- `-c, --collection TEXT` — collection name (default: `default`)
- `-e, --engine TEXT` — engine name (default: `fitz_krag`)
- `--chat` — interactive multi-turn mode
- `--endpoint TEXT` — OpenAI-compatible URL; overrides `chat_base_url`
- `-m, --model TEXT` — chat model name; overrides smart-tier model
- `--api-key-env TEXT` — env var holding the API key; overrides `chat_api_key_env`

**Chat mode**
- Type questions naturally.
- Exit with `exit`, `quit`, or `Ctrl+C`.

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
chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: http://localhost:8080/v1
chat_smart_model: qwen2.5-7b-instruct
collection: default
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
# Start a local OpenAI-compatible server
llama-server -m qwen2.5-7b-instruct.gguf --port 8080 &

# Ingest + query (config is auto-created on first run)
fitz query "What's in my docs?" --source ./docs
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
fitz query "What is X?" \
  --endpoint https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY \
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
