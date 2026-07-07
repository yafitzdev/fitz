# docs/API.md
# REST API Reference

Complete reference for the Fitz REST API.

---

## Quick Start

```bash
# Install with API support
pip install fitz-sage[api]

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

# All interfaces (for Docker/remote access)
fitz serve --host 0.0.0.0

# Development mode
fitz serve --reload
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Query knowledge base and return an answer |
| POST | `/evidence` | Retrieve governed evidence without answer synthesis |
| POST | `/chat` | Multi-turn chat |
| GET | `/collections` | List collections |
| GET | `/collections/{name}` | Get collection stats |
| POST | `/collections/{name}/documents` | Ingest documents (background indexing) |
| GET | `/collections/{name}/status` | Indexing progress |
| DELETE | `/collections/{name}` | Delete collection |
| GET | `/health` | Health check |

---

## POST /query

Query the knowledge base with a single question and return answer text plus
source attribution.

### Request

```json
{
  "question": "What is the refund policy?",
  "source": "./docs",
  "collection": "default"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `question` | string | Yes | - | The question to ask |
| `source` | string | No | null | Path to file or directory. If provided, registers documents before querying. |
| `collection` | string | No | `"default"` | Collection to query |

### Response

```json
{
  "text": "The refund policy allows returns within 30 days...",
  "mode": "trustworthy",
  "sources": [
    {
      "source_id": "policies/refund.md",
      "excerpt": "Returns are accepted within 30 days of purchase...",
      "metadata": {
        "chunk_index": 2,
        "page": 1
      }
    }
  ],
  "metadata": {}
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The answer text |
| `mode` | string | Runtime mode: `trustworthy`, `disputed`, or `abstain` |
| `sources` | array | Source attribution for the answer |
| `metadata` | object | Extra answer metadata; on runtime `abstain`, includes `gap_context` (what's missing and what to add) |

### Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

---

## POST /evidence

Retrieve a governed `EvidencePack` without answer synthesis. This is the REST
equivalent of `fitz query`, `fitz retrieve`, and `fitz_sage.evidence()`.

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
| `source` | string | No | null | Path to file or directory. If provided, registers documents before querying. |
| `collection` | string | No | `"default"` | Collection to query |
| `conversation_history` | array | No | `[]` | Optional chat history for query rewriting |

### Response

```json
{
  "query": "What is the refund policy?",
  "mode": "trustworthy",
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

The server is **stateless** - the client must manage and send conversation history.

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

Same as `/query`:

```json
{
  "text": "For returns, you need to...",
  "mode": "trustworthy",
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

### Response

```json
[
  {"name": "default", "chunk_count": 234},
  {"name": "physics", "chunk_count": 567}
]
```

### Example

```bash
curl http://localhost:8000/collections
```

---

## GET /collections/{name}

Get statistics for a specific collection.

### Response

```json
{
  "name": "default",
  "chunk_count": 234,
  "metadata": {
    "created_at": "2024-01-15T10:30:00",
    "last_updated": "2024-01-16T14:20:00"
  }
}
```

### Example

```bash
curl http://localhost:8000/collections/default
```

---

## POST /collections/{name}/documents

Register documents into a collection. Indexing runs in the background — queries
work immediately and improve as it completes. Returns `202 Accepted` with the
current indexing status; poll `GET /collections/{name}/status` for progress.

### Request

```json
{"source": "./docs"}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | Path to a file or directory to ingest |

### Response (`202 Accepted`)

```json
{"total": 42, "indexed": 0, "pending": 42, "complete": false, "by_state": {"registered": 42}}
```

### Example

```bash
curl -X POST http://localhost:8000/collections/default/documents \
  -H "Content-Type: application/json" \
  -d '{"source": "./docs"}'
```

---

## GET /collections/{name}/status

Background-indexing progress for a collection.

### Response

```json
{"total": 42, "indexed": 30, "pending": 12, "complete": false, "by_state": {"enriched": 30, "parsed": 12}}
```

| Field | Type | Description |
|-------|------|-------------|
| `total` | integer | Files registered |
| `indexed` | integer | Files indexed (enriched or summarized) |
| `pending` | integer | Files still pending (registered or parsed) |
| `complete` | boolean | True when no files remain pending |
| `by_state` | object | File counts per indexing state |

### Example

```bash
curl http://localhost:8000/collections/default/status
```

---

## DELETE /collections/{name}

Delete a collection and all its chunks.

### Response

```json
{
  "deleted": true,
  "collection": "default",
  "chunks_deleted": 234
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
  "config_exists": true
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
| 404 | Resource not found |
| 500 | Internal server error |
| 501 | Feature not implemented |

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
| `trustworthy` | Runtime mode for sufficient evidence | Clear, unambiguous sources |
| `disputed` | Runtime mode for disputed evidence | Sources disagree |
| `abstain` | Runtime mode for insufficient evidence | Missing or incomplete evidence |

---

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Query
response = requests.post(f"{BASE_URL}/query", json={
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
  const response = await fetch(`${BASE_URL}/query`, {
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
