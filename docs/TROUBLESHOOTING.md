# Troubleshooting Guide

Common issues and solutions for fitz-sage **v0.12.0+** (single OpenAI-compatible
HTTP protocol, SQLite + FTS5 storage, no embeddings, no vector DB).

---

## Quick Diagnostics

Open the config file directly at `.fitz/config.yaml` and verify the
chat endpoint URL, API-key environment variable, and collection name
are correct.

---

## Common Issues

### Config Not Found

**Error:**
```
ConfigNotFoundError: Config file not found
```

**Solution:**

The config is created automatically on first run — edit it, or pass
`--endpoint`, `--model`, and `--api-key-env` directly on the
`fitz query` command:

```bash
fitz query "What is X?" \
  --endpoint https://api.openai.com/v1 \
  --model gpt-4o-mini \
  --api-key-env OPENAI_API_KEY \
  --source ./docs
```

---

### Cannot connect to chat endpoint

**Error:**
```
LLMError: Cannot connect to http://localhost:8080/v1
```

**Solution:**

1. Confirm an OpenAI-compatible server is running and reachable at the
   configured `chat_base_url`. Common local options:

   ```bash
   # llama.cpp
   llama-server -m model.gguf --port 8080

   # vLLM
   python -m vllm.entrypoints.openai.api_server --model my-model --port 8080

   # LM Studio: enable the local server in Settings → Developer
   ```

2. Verify reachability:
   ```bash
   curl http://localhost:8080/v1/models
   ```

3. If using Ollama, point fitz-sage at Ollama's OpenAI-compatible
   endpoint (`http://localhost:11434/v1`). The legacy `ollama`
   provider name was removed in v0.12.0 — use `endpoint` instead.

---

### No API Key

**Error:**
```
AuthenticationError: API key not found in environment variable OPENAI_API_KEY
```

**Solution:**

Set the env var named in `chat_api_key_env` (defaults vary by config):

```bash
# Linux / macOS
export OPENAI_API_KEY="..."
export TOGETHER_API_KEY="..."

# Windows (PowerShell)
$env:OPENAI_API_KEY = "..."

# Windows (cmd)
set OPENAI_API_KEY=...
```

For unauthenticated local servers, leave `chat_api_key_env` unset.

---

### Storage path errors

**Error:**
```
sqlite3.OperationalError: unable to open database file
```

**Cause:** The workspace storage directory doesn't exist or isn't writable.

**Solution:**

1. Confirm `<workspace>/sqlite/` exists. By default `<workspace>` is
   `~/.fitz/`.
2. Verify write permissions for the user running fitz-sage.
3. To start fresh, delete the per-collection file (the collection
   name maps to `fitz_<collection>.db`):
   ```bash
   rm ~/.fitz/sqlite/fitz_<collection>.db
   rm ~/.fitz/sqlite/fitz_<collection>.db-wal
   rm ~/.fitz/sqlite/fitz_<collection>.db-shm
   ```

---

### Stale collection / schema mismatch

If a collection was created on a much older fitz-sage and a `SELECT`
errors out with a missing column, the simplest path is to delete the
collection's `.db` and re-ingest. There is no in-place migration step.

```bash
fitz collections delete my_collection
fitz query --source ./docs "..."
```

---

### Rate Limit Error

**Error:**
```
RateLimitError: Rate limit exceeded
```

**Solution:**

1. Wait and retry (the chat provider applies exponential backoff
   automatically — see `fitz_sage/llm/auth/`).
2. Point `chat_fast` at a cheaper model for the bulk of the work:
   ```yaml
   chat_fast: gpt-4o-mini
   chat_balanced: gpt-4o-mini
   chat_smart: gpt-4o
   ```

---

### Empty Chunks

**Error:**
```
ValueError: No chunks created from documents
```

**Causes:**
- Documents are empty or unreadable.
- Parser failed silently (enable DEBUG logging to see why).
- All content filtered out by chunking rules.

**Solution:**

1. Check document contents manually.
2. Enable DEBUG logging by setting `log_level: DEBUG` in
   `.fitz/config.yaml`, then re-run the query.
3. Check supported formats in [INGESTION.md](INGESTION.md).

---

### No Documents Found

**Error:**
```
ValueError: No documents found in ./path
```

**Causes:**
- Wrong path.
- No supported file types.
- Files filtered by `.gitignore` patterns.

**Solution:**

1. Verify path exists.
2. Check file extensions (supported: `.pdf`, `.docx`, `.md`, `.txt`,
   `.py`, `.go`, `.ts`, `.java`, `.cs`, `.sql`, `.xlsx`, `.csv`, etc.).
3. Try with a specific file:
   ```bash
   fitz query --source ./path/file.pdf "test query"
   ```

---

### Timeout Errors

**Error:**
```
TimeoutError: Request timed out
```

**Solution:**

1. Check network connection.
2. For large documents, lower `chunk_size` or `top_addresses` so
   each LLM call stays within timeout.
3. Switch to a faster model for the synthesizer.

---

## Debugging

### Enable Debug Logging

```yaml
# In ~/.fitz/config/fitz_krag.yaml
log_level: DEBUG
```

### Inspect State File

```bash
cat .fitz/ingest_state.json | python -m json.tool
```

### Test Chat Endpoint Directly

```python
from fitz_sage.llm.client import get_chat

chat = get_chat("endpoint", tier="smart")  # or use the full URL form
response = chat.chat([{"role": "user", "content": "Hello"}])
print(response)
```

### Inspect the SQLite Store

```bash
sqlite3 ~/.fitz/sqlite/fitz_<collection>.db
.tables
SELECT COUNT(*) FROM krag_sections;
SELECT id, title FROM krag_sections LIMIT 5;
```

---

## Error Reference

### Exception Hierarchy

```
EngineError (base)
├── ConfigurationError       # Config issues
├── QueryError               # Invalid query
├── KnowledgeError           # Retrieval issues
├── GenerationError          # LLM issues
├── TimeoutError             # Timeout
└── UnsupportedOperationError

APIError
├── AuthenticationError      # Bad API key
├── RateLimitError           # Rate limited
└── ModelNotFoundError       # Invalid model

ConfigError
├── ConfigNotFoundError      # Missing config
├── ConfigParseError         # Invalid YAML
└── ConfigValidationError    # Schema error
```

### HTTP Status Codes (REST API)

| Code | Meaning |
|------|---------|
| 400  | Bad request (invalid input) |
| 401  | Authentication failed |
| 404  | Collection / resource not found |
| 429  | Rate limited |
| 500  | Internal server error |
| 501  | Feature not supported |

---

## Getting Help

1. **Check config:** inspect `.fitz/config.yaml` directly
2. **Check logs:** enable DEBUG level
3. **Report issues:** [GitHub Issues](https://github.com/yafitzdev/fitz-sage/issues)

When reporting, include:
- fitz-sage version: `pip show fitz-sage`
- Python version: `python --version`
- OS
- Full traceback
- Effective config (the contents of `.fitz/config.yaml`, with secrets redacted)

---

## See Also

- [CONFIG.md](CONFIG.md) — Configuration reference
- [CLI.md](CLI.md) — CLI commands
- [INGESTION.md](INGESTION.md) — Ingestion pipeline
- [features/platform/unified-storage.md](features/platform/unified-storage.md) — SQLite + FTS5 storage layer
