# Plugin Development Guide

Fitz uses a YAML-based plugin system for LLM providers and vector databases. This guide explains how to create custom plugins.

---

## Overview

Plugins are YAML files that describe how to interact with external services. No Python code required.

**Plugin Types:**

| Type | Location | Purpose |
|------|----------|---------|
| Chat | `fitz_ai/llm/chat/` | LLM chat/completion |
| Embedding | `fitz_ai/llm/embedding/` | Text embeddings |
| Rerank | `fitz_ai/llm/rerank/` | Document reranking |
| Vector DB | `fitz_ai/vector_db/plugins/` | Vector storage |

---

## LLM Plugins

### Chat Plugin

Creates a chat/completion provider.

**File:** `fitz_ai/llm/chat/my_provider.yaml`

```yaml
# =============================================================================
# IDENTITY
# =============================================================================
plugin_name: "my_provider"
plugin_type: "chat"
version: "1.0"

# =============================================================================
# PROVIDER
# =============================================================================
provider:
  name: "my_provider"
  base_url: "https://api.myprovider.com/v1"

# =============================================================================
# AUTHENTICATION
# =============================================================================
auth:
  type: "bearer"                    # bearer, header, query, none
  header_name: "Authorization"
  header_format: "Bearer {key}"
  env_vars:
    - "MY_PROVIDER_API_KEY"

# =============================================================================
# ENDPOINT
# =============================================================================
endpoint:
  path: "/chat/completions"
  method: "POST"
  timeout: 120

# =============================================================================
# DEFAULTS
# =============================================================================
# Model tiers for different use cases:
#   smart: Best quality (user-facing queries)
#   fast: Best speed (background tasks, enrichment)
#   balanced: Cost-effective (bulk operations)
defaults:
  models:
    smart: "my-model-large"
    fast: "my-model-small"
    balanced: "my-model-medium"
  temperature: 0.2
  max_tokens: null

# =============================================================================
# REQUEST CONFIGURATION
# =============================================================================
request:
  # Message format transformation
  messages_transform: "openai_chat"   # openai_chat, cohere_chat, anthropic_chat, ollama_chat

  # Static fields added to every request
  static_fields:
    stream: false

  # Map our params to API params
  param_map:
    model: "model"
    temperature: "temperature"
    max_tokens: "max_tokens"

# =============================================================================
# RESPONSE CONFIGURATION
# =============================================================================
response:
  # Path to extract response text (dot notation, array indexing supported)
  content_path: "choices[0].message.content"
  is_array: false
  array_index: 0

  # Optional metadata extraction
  metadata_paths:
    finish_reason: "choices[0].finish_reason"
    tokens_input: "usage.prompt_tokens"
    tokens_output: "usage.completion_tokens"
```

**Message Transforms:**

| Transform | Format |
|-----------|--------|
| `openai_chat` | `{"messages": [{"role": "user", "content": "..."}]}` |
| `cohere_chat` | Cohere v2 format |
| `anthropic_chat` | Anthropic format |
| `ollama_chat` | Ollama format |

---

### Embedding Plugin

Creates an embedding provider.

**File:** `fitz_ai/llm/embedding/my_provider.yaml`

```yaml
plugin_name: "my_provider"
plugin_type: "embedding"
version: "1.0"

provider:
  name: "my_provider"
  base_url: "https://api.myprovider.com/v1"

auth:
  type: "bearer"
  header_name: "Authorization"
  header_format: "Bearer {key}"
  env_vars:
    - "MY_PROVIDER_API_KEY"

endpoint:
  path: "/embeddings"
  method: "POST"
  timeout: 30

defaults:
  model: "my-embed-model"

request:
  # Field name for input text
  input_field: "input"

  # How to wrap input: list, string, object
  input_wrap: "list"

  static_fields: {}

  param_map:
    model: "model"

response:
  # Path to embedding vector(s)
  embeddings_path: "data[0].embedding"
  is_array: true
  array_index: 0
```

---

### Rerank Plugin

Creates a reranking provider.

**File:** `fitz_ai/llm/rerank/my_provider.yaml`

```yaml
plugin_name: "my_provider"
plugin_type: "rerank"
version: "1.0"

provider:
  name: "my_provider"
  base_url: "https://api.myprovider.com/v1"

auth:
  type: "bearer"
  header_name: "Authorization"
  header_format: "Bearer {key}"
  env_vars:
    - "MY_PROVIDER_API_KEY"

endpoint:
  path: "/rerank"
  method: "POST"
  timeout: 30

defaults:
  model: "my-rerank-model"
  top_n: 5

request:
  query_field: "query"
  documents_field: "documents"

  static_fields: {}

  param_map:
    model: "model"
    top_n: "top_n"

response:
  results_path: "results"
  result_index_path: "index"
  result_score_path: "relevance_score"
```

---

## Vector DB Plugins

Creates a vector database provider.

**File:** `fitz_ai/vector_db/plugins/my_vectordb.yaml`

```yaml
name: my_vectordb
type: vector_db
description: My custom vector database

# =============================================================================
# Features
# =============================================================================
features:
  # If true, string IDs are converted to UUIDs
  requires_uuid_ids: false

  # Auto-detection function (optional)
  auto_detect: null

  # Whether DB uses namespaces vs collections
  supports_namespaces: false

# =============================================================================
# Connection
# =============================================================================
connection:
  type: http
  base_url: "http://{{host}}:{{port}}"
  default_host: localhost
  default_port: 8000

  auth:
    type: bearer
    env_var: MY_VECTORDB_API_KEY
    header: Authorization
    scheme: "Bearer"
    optional: true

# =============================================================================
# Operations
# =============================================================================
operations:
  # Search for similar vectors
  search:
    endpoint: /collections/{{collection}}/search
    method: POST
    body:
      vector: "{{query_vector}}"
      limit: "{{limit}}"
    response:
      results_path: results
      mapping:
        id: id
        score: score
        payload: metadata

  # Insert/update vectors
  upsert:
    endpoint: /collections/{{collection}}/points
    method: PUT

    # Auto-create collection on 404
    auto_create_collection: true
    create_collection_endpoint: /collections/{{collection}}
    create_collection_method: PUT
    create_collection_body:
      dimension: "{{vector_dim}}"

    body:
      points: "{{points}}"

  # Count points
  count:
    endpoint: /collections/{{collection}}/count
    method: GET
    response:
      count_path: count

  # Delete collection
  delete_collection:
    endpoint: /collections/{{collection}}
    method: DELETE
    response:
      success_codes: [200, 404]

  # List collections
  list_collections:
    endpoint: /collections
    method: GET
    response:
      collections_path: collections
      name_field: name
```

---

## Using Your Plugin

Once created, plugins are auto-discovered:

```python
from fitz_ai.llm.registry import get_llm_plugin, available_llm_plugins
from fitz_ai.vector_db.registry import get_vector_db_plugin, available_vector_db_plugins

# List available plugins
print(available_llm_plugins("chat"))      # ['cohere', 'openai', 'my_provider', ...]
print(available_vector_db_plugins())       # ['qdrant', 'my_vectordb', ...]

# Use your plugin
chat = get_llm_plugin(plugin_name="my_provider", plugin_type="chat")
response = chat.chat([{"role": "user", "content": "Hello"}])

embedder = get_llm_plugin(plugin_name="my_provider", plugin_type="embedding")
vector = embedder.embed("Some text")

vectordb = get_vector_db_plugin("my_vectordb", host="localhost", port=8000)
results = vectordb.search(collection="test", vector=[0.1, 0.2, ...], limit=5)
```

In configuration:

```yaml
# fitz.yaml
chat:
  plugin_name: my_provider
  kwargs:
    model: my-model-large

embedding:
  plugin_name: my_provider

vector_db:
  plugin_name: my_vectordb
  kwargs:
    host: localhost
    port: 8000
```

---

## Path Notation

Response paths use dot notation with array indexing:

| Path | Accesses |
|------|----------|
| `data` | `response["data"]` |
| `data.text` | `response["data"]["text"]` |
| `data[0]` | `response["data"][0]` |
| `data[0].text` | `response["data"][0]["text"]` |
| `choices[0].message.content` | `response["choices"][0]["message"]["content"]` |

---

## Authentication Types

| Type | Description |
|------|-------------|
| `bearer` | `Authorization: Bearer {key}` |
| `header` | Custom header with key |
| `query` | API key as query parameter |
| `none` | No authentication |

---

## Model Tiers

Chat plugins support three model tiers for cost optimization:

| Tier | Use Case | Example |
|------|----------|---------|
| `smart` | User-facing queries | `gpt-4`, `command-a-03-2025` |
| `fast` | Background tasks, enrichment | `gpt-4o-mini`, `command-r7b` |
| `balanced` | Bulk operations, evaluation | Middle-ground models |

```python
# Use smart tier for queries (default)
chat = get_llm_plugin(plugin_name="cohere", plugin_type="chat", tier="smart")

# Use fast tier for background enrichment
chat = get_llm_plugin(plugin_name="cohere", plugin_type="chat", tier="fast")
```

---

## Existing Plugins

### Chat Plugins

| Plugin | Provider | Models |
|--------|----------|--------|
| `cohere` | Cohere | command-a-03-2025, command-r-plus |
| `openai` | OpenAI | gpt-4o, gpt-4o-mini |
| `anthropic` | Anthropic | claude-3-opus, claude-3-sonnet |
| `azure_openai` | Azure | Deployed OpenAI models |
| `local_ollama` | Ollama | llama3.2, mistral, etc. |

### Embedding Plugins

| Plugin | Provider | Models |
|--------|----------|--------|
| `cohere` | Cohere | embed-english-v3.0 |
| `openai` | OpenAI | text-embedding-3-small |
| `azure_openai` | Azure | Deployed embedding models |
| `local_ollama` | Ollama | nomic-embed-text, etc. |

### Vector DB Plugins

| Plugin | Provider |
|--------|----------|
| `qdrant` | Qdrant (local or cloud) |
| `pinecone` | Pinecone |
| `weaviate` | Weaviate |
| `milvus` | Milvus |
| `local_faiss` | FAISS (local, no server) |

---

## Troubleshooting

### Plugin Not Found

```
LLMRegistryError: Unknown chat plugin: 'my_provider'
```

- Check file is in correct directory (`fitz_ai/llm/chat/my_provider.yaml`)
- Check `plugin_name` matches filename (without `.yaml`)

### Validation Error

```
YAMLPluginValidationError: Invalid YAML plugin
  - provider -> base_url: must start with http://
```

- Check YAML syntax
- Validate against schema requirements
- Use existing plugins as templates

### Authentication Failed

- Verify environment variable is set: `echo $MY_PROVIDER_API_KEY`
- Check `auth.env_vars` lists the correct variable name
- Verify `auth.header_format` matches provider requirements

### Response Extraction Failed

- Test API manually to see actual response structure
- Update `response.content_path` to match actual path
- Use array notation `[0]` if response is wrapped in array
