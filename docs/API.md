# docs/API.md
# REST API Reference

Complete reference for the Fitz REST API.

---

## Quick Start

```bash
# Install with API support
pip install fitz-sage[api]

# Initialize the workspace and a collection once
fitz retrieve "What is indexed?" --source ./docs

# Start the server
fitz serve

# Server runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

---

## Server Options

```bash
fitz serve [OPTIONS]

Options:
  -p, --port INTEGER    Port number (default: 8000)
  --host TEXT          Host to bind (default: 127.0.0.1)
  --reload             Enable auto-reload for development
```

**Examples:**

```bash
# Custom port
fitz serve -p 3000

# All interfaces (requires FITZ_API_KEY)
export FITZ_API_KEY="replace-with-a-random-secret"
fitz serve --host 0.0.0.0

# Development mode
fitz serve --reload
```

---

## Security Boundary

Loopback clients (`127.0.0.1` and `::1`) may call the API without a key. Any
non-loopback client must send the key configured in `FITZ_API_KEY`:

```bash
curl -H "X-Fitz-API-Key: $FITZ_API_KEY" http://server:8000/health
```

Browser cross-origin access is disabled by default. Set a comma-separated
allowlist only for origins you control:

```bash
export FITZ_API_ALLOWED_ORIGINS="https://app.example.com,http://localhost:3000"
```

Request `source` paths are server-local paths. They must resolve inside the
server's current working directory by default. `FITZ_API_SOURCE_ROOTS` can set
an explicit platform-path-separated allowlist of roots.

Collection names must match `[a-z0-9][a-z0-9_-]{0,63}`. fitz-sage does not
normalize explicit names, so `project-a` and `project_a` remain distinct.

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/answer` | Retrieve evidence and synthesize an answer |
| POST | `/evidence` | Retrieve governed evidence without answer synthesis |
| POST | `/chat` | Multi-turn chat |
| GET | `/collections` | List collections |
| GET | `/collections/{name}` | Get collection stats |
| POST | `/collections/{name}/documents` | Build/update the searchable source index |
| GET | `/collections/{name}/status` | Source-index and enrichment status |
| DELETE | `/collections/{name}` | Delete collection |
| GET | `/health` | Health check |

---

## POST /answer

Query the knowledge base with a single question and return answer text plus
source attribution. The selected collection config must include a
`synthesizer`; use `/evidence` for retrieval without generation.

### Request

```json
{
  "question": "What is the refund policy?",
  "source": "./docs",
  "collection": "default",
  "conversation_history": []
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to ask |
| `source` | string | No | null | Allowed server-local file or directory. If provided, registers it and waits until the query surface is ready. |
| `collection` | string | No | `"default"` | Collection to query |
| `conversation_history` | array | No | `[]` | History made available to configured query intelligence; otherwise retrieval uses the current question. |

### Response

```json
{
  "text": "The refund policy allows returns within 30 days...",
  "mode": "sufficient",
  "sources": [
    {
      "source_id": "policies/refund.md",
      "excerpt": "Returns are accepted within 30 days of purchase...",
      "metadata": {
        "kind": "section",
        "location": "Refund Policy",
        "file_path": "policies/refund.md",
        "line_range": [4, 18]
      }
    }
  ],
  "metadata": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The answer text |
| `mode` | string | Runtime mode: `sufficient`, `disputed`, or `insufficient` |
| `sources` | array | Source attribution for the answer |
| `metadata` | object | Extra answer metadata; on runtime `insufficient`, includes `gap_context` (what's missing and what to add) |

### Example

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

---

## POST /evidence

Retrieve a governed `EvidencePack` without answer synthesis. This is the REST
equivalent of `fitz retrieve` and `fitz_sage.evidence()`.

### Request

```json
{
  "question": "What is the refund policy?",
  "source": "./docs",
  "collection": "default",
  "conversation_history": []
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to retrieve evidence for |
| `source` | string | No | null | Allowed server-local file or directory. If provided, registers it and waits until the query surface is ready. |
| `collection` | string | No | `"default"` | Collection to query |
| `conversation_history` | array | No | `[]` | History made available to configured query intelligence; otherwise retrieval uses the current question. |

### Response

```json
{
  "query": "What is the refund policy?",
  "mode": "sufficient",
  "items": [
    {
      "rank": 1,
      "source_id": "policies/refund.md",
      "file_path": "policies/refund.md",
      "address_kind": "section",
      "address_location": "Refund Policy",
      "line_range": [4, 18],
      "score": 0.91,
      "excerpt": "Returns are accepted within 30 days of purchase...",
      "content": "Returns are accepted within 30 days of purchase...",
      "metadata": {}
    }
  ],
  "reasons": ["Pyrrho: sources support a confident answer."],
  "timings": {},
  "indexing_status": {"complete": true},
  "metadata": {}
}
```

See [EVIDENCE_PACK.md](EVIDENCE_PACK.md) for field meanings and governance
metadata.

### Example

```bash
curl -X POST http://localhost:8000/evidence \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "source": "./docs"}'
```

---

## POST /chat

Multi-turn conversation with the knowledge base.

Like `/answer`, this endpoint requires a configured `synthesizer`.

The server is **stateless** - the client must manage and send conversation
history. A configured `query_intelligence` provider can use that history to
rewrite conversational references; the deterministic default does not claim
automatic pronoun resolution.

### Request

```json
{
  "message": "What about returns?",
  "history": [
    {"role": "user", "content": "What is the refund policy?"},
    {"role": "assistant", "content": "The refund policy allows returns within 30 days..."}
  ],
  "collection": "default"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | Current user message |
| `history` | array | No | `[]` | Previous messages |
| `collection` | string | No | `"default"` | Collection to query |

**History message format:**

```json
{"role": "user" | "assistant", "content": "message text"}
```

### Response

Same as `/answer`:

```json
{
  "text": "For returns, you need to...",
  "mode": "sufficient",
  "sources": [...]
}
```

### Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about returns?",
    "history": [
      {"role": "user", "content": "What is the refund policy?"},
      {"role": "assistant", "content": "The refund policy allows..."}
    ]
  }'
```

---

## GET /collections

List all available collections.

`item_count` is the number of persisted document-section rows. It is not a
combined count of sections, code symbols, and native table rows.

### Response

```json
[
  {"name": "default", "item_count": 234},
  {"name": "physics", "item_count": 567}
]
```

### Example

```bash
curl http://localhost:8000/collections
```

---

## GET /collections/{name}

Get statistics for a specific collection.

`item_count` has the same document-section-row meaning as the list endpoint.

### Response

```json
{
  "name": "default",
  "item_count": 234,
  "metadata": {}
}
```

### Example

```bash
curl http://localhost:8000/collections/default
```

---

## POST /collections/{name}/documents

Register documents into a collection. The request returns after supported files
are stored in the searchable source index. Optional model-backed enrichment may
continue afterward.

### Request

```json
{"source": "./docs"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Allowed server-local file or directory to ingest |

### Response (`200 OK`)

```json
{
  "discovered": 45,
  "total": 42,
  "indexed": 42,
  "pending": 0,
  "failed": 0,
  "failed_files": [],
  "unsupported": 3,
  "unsupported_files": [
    {"path": "archive.bin", "extension": ".bin"}
  ],
  "healthy": true,
  "complete": true,
  "query_ready": true,
  "by_index_state": {"indexed": 42},
  "enrichment": {
    "total": 42,
    "completed": 0,
    "pending": 42,
    "failed": 0,
    "failed_files": [],
    "pending_files": [
      {"path": "guide.pdf", "state": "pending", "priority": 4}
    ],
    "finalization": "pending",
    "finalization_error": null,
    "complete": false
  },
  "by_enrichment_state": {"pending": 42}
}
```

### Example

```bash
curl -X POST http://localhost:8000/collections/default/documents \
  -H "Content-Type: application/json" \
  -d '{"source": "./docs"}'
```

---

## GET /collections/{name}/status

Source-index health and optional enrichment progress for a collection.

### Response

```json
{
  "discovered": 45,
  "total": 42,
  "indexed": 42,
  "pending": 0,
  "failed": 0,
  "failed_files": [],
  "unsupported": 3,
  "unsupported_files": [
    {"path": "archive.bin", "extension": ".bin"}
  ],
  "healthy": true,
  "complete": true,
  "query_ready": true,
  "by_index_state": {"indexed": 42},
  "enrichment": {
    "total": 42,
    "completed": 30,
    "pending": 12,
    "failed": 0,
    "failed_files": [],
    "pending_files": [
      {"path": "guide.pdf", "state": "pending", "priority": 4}
    ],
    "finalization": "pending",
    "finalization_error": null,
    "complete": false
  },
  "by_enrichment_state": {"complete": 30, "pending": 12}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `discovered` | integer | All files found under the registered source |
| `total` | integer | Supported files, including indexing failures |
| `indexed` | integer | Files stored in the searchable source index |
| `pending` | integer | Files not yet settled by source indexing |
| `failed` | integer | Supported files that failed source indexing |
| `failed_files` | array | Source-index failures with path, stage, and error |
| `unsupported` | integer | Files outside the enabled format contract |
| `unsupported_files` | array | Unsupported paths and extensions |
| `healthy` | boolean | True when no supported file failed indexing |
| `complete` | boolean | True when every supported file indexed successfully |
| `query_ready` | boolean | True when no supported file remains pending |
| `by_index_state` | object | File counts per source-index state |
| `enrichment` | object | Independent entity/hierarchy progress and failures |
| `by_enrichment_state` | object | Indexed-file counts per enrichment state |

### Example

```bash
curl http://localhost:8000/collections/default/status
```

---

## DELETE /collections/{name}

Delete a collection and its manifest and SQLite data.

### Response

```json
{
  "deleted": true,
  "collection": "default"
}
```

### Example

```bash
curl -X DELETE http://localhost:8000/collections/old_data
```

---

## GET /health

Health check endpoint.

### Response

```json
{
  "status": "healthy",
  "version": "<installed version>",
  "components": {"sqlite": true}
}
```

### Example

```bash
curl http://localhost:8000/health
```

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 400 | Bad request (invalid input) |
| 401 | Invalid or missing API key for remote access |
| 403 | Remote access is not configured |
| 404 | Resource not found |
| 422 | Request body failed schema validation |
| 500 | Internal server error |

**Error response format:**

```json
{
  "detail": "Error message here"
}
```

---

## Runtime Answer Modes

The `mode` field is the runtime API value. Pyrrho v2's native model verdict is
available in governance metadata as `evidence_verdict` with
`SUFFICIENT`, `DISPUTED`, or `INSUFFICIENT` semantics.

| Mode | Description | Typical Cause |
|------|-------------|---------------|
| `sufficient` | Runtime mode for sufficient evidence | Clear, unambiguous sources |
| `disputed` | Runtime mode for disputed evidence | Sources disagree |
| `insufficient` | Runtime mode for insufficient evidence | Missing or incomplete evidence |

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Query
response = requests.post(f"{BASE_URL}/answer", json={
    "question": "What is the refund policy?",
    "collection": "default"
})
answer = response.json()
print(answer["text"])

# Chat with history
history = []
message = "What is the refund policy?"

response = requests.post(f"{BASE_URL}/chat", json={
    "message": message,
    "history": history
})
answer = response.json()

# Update history for next turn
history.append({"role": "user", "content": message})
history.append({"role": "assistant", "content": answer["text"]})

# Continue conversation
response = requests.post(f"{BASE_URL}/chat", json={
    "message": "What about returns?",
    "history": history
})
```

---

## JavaScript Client Example

```javascript
const BASE_URL = 'http://localhost:8000';

// Query
async function query(question) {
  const response = await fetch(`${BASE_URL}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  return response.json();
}

// Chat
async function chat(message, history = []) {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history })
  });
  return response.json();
}

// Usage
const answer = await query("What is the refund policy?");
console.log(answer.text);
```

---

## See Also

- [SDK.md](SDK.md) - Python SDK documentation
- [CLI.md](CLI.md) - CLI reference
- [CONFIG.md](CONFIG.md) - Configuration reference
