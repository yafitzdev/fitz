<!-- docs/features/platform/enterprise-gateway.md -->
# Enterprise LLM Gateway

## Problem

Enterprise deployments often can't use direct LLM provider APIs:

- **Corporate proxy/gateway** - LLM traffic must route through an internal gateway
- **OAuth2 authentication** - Machine-to-machine (M2M) auth instead of API keys
- **Custom CA certificates** - Internal PKI, not public certificate authorities
- **mTLS requirements** - Mutual TLS with client certificates
- **Multiple auth headers** - Bearer token for gateway + API key for underlying LLM

Standard provider SDKs don't support these enterprise patterns.

## Solution: Enterprise Gateway Provider

Fitz includes a lightweight, SDK-free enterprise provider that supports:

```
┌──────────────────────────────────────────────────────────────────┐
│  Your Application (Fitz)                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                    OAuth2 Client Credentials
                    + Bearer Token
                    + LLM API Key Header
                    + Custom CA Cert
                    + mTLS (optional)
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Enterprise LLM Gateway                                          │
│  (OpenAI-compatible API)                                         │
│  https://llm.corp.internal/v1                                    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Underlying LLM Provider                                         │
│  (OpenAI, Azure, Anthropic, etc.)                                │
└──────────────────────────────────────────────────────────────────┘
```

## Configuration

### Basic M2M Authentication

```yaml
# ~/.fitz/config/fitz_krag.yaml
synthesizer: enterprise/openai/gpt-4o
chat_base_url: https://llm.corp.internal/v1

auth:
  type: m2m
  token_url: https://auth.corp.internal/oauth/token
  client_id: ${CLIENT_ID}
  client_secret: ${CLIENT_SECRET}
  scope: llm.access  # optional
```

### Enterprise Auth (M2M + API Key)

Some gateways require both OAuth2 bearer token AND an LLM API key:

```yaml
synthesizer: enterprise/openai/gpt-4o
chat_base_url: https://llm.corp.internal/v1

auth:
  type: enterprise
  # OAuth2 M2M for gateway authentication
  token_url: https://auth.corp.internal/oauth/token
  client_id: ${CLIENT_ID}
  client_secret: ${CLIENT_SECRET}
  scope: llm.access
  # LLM API key for underlying provider
  llm_api_key_env: CORP_LLM_API_KEY
  llm_api_key_header: X-Api-Key  # default
```

This sends two headers:
- `Authorization: Bearer <oauth_token>` (for gateway)
- `X-Api-Key: <llm_api_key>` (for underlying LLM)

### Custom CA Certificate

For internal PKI (non-public certificate authorities):

```yaml
synthesizer: enterprise/openai/gpt-4o
chat_base_url: https://llm.corp.internal/v1

auth:
  type: m2m
  token_url: https://auth.corp.internal/oauth/token
  client_id: ${CLIENT_ID}
  client_secret: ${CLIENT_SECRET}
  cert_path: /etc/ssl/corp-ca-bundle.crt
```

### Mutual TLS (mTLS)

For gateways requiring client certificate authentication:

```yaml
synthesizer: enterprise/openai/gpt-4o
chat_base_url: https://llm.corp.internal/v1

auth:
  type: m2m
  token_url: https://auth.corp.internal/oauth/token
  client_id: ${CLIENT_ID}
  client_secret: ${CLIENT_SECRET}
  cert_path: /etc/ssl/corp-ca-bundle.crt
  # mTLS client certificate
  client_cert_path: /etc/ssl/client.crt
  client_key_path: /etc/ssl/client.key
  client_key_password: ${KEY_PASSWORD}  # if key is encrypted
```

## Environment Variable Resolution

Secrets should never be in config files. Use `${VAR}` syntax:

```yaml
auth:
  client_id: ${CORP_CLIENT_ID}        # Reads from environment
  client_secret: ${CORP_CLIENT_SECRET}
  client_key_password: ${KEY_PASSWORD}
```

At startup, Fitz resolves these to actual values from the environment. Missing variables cause immediate failure with clear error messages.

## How It Works

### OAuth2 Token Management

The M2MAuth provider handles the OAuth2 client credentials flow:

1. **Initial request** - Fetches access token from token endpoint
2. **Token caching** - Caches token until near expiry (60s margin)
3. **Auto-refresh** - Transparently refreshes token before expiry
4. **Thread-safe** - Multiple concurrent requests share the same token

### Resilience Features

- **Exponential backoff** - Retries transient failures (timeout, network) with 1s → 2s → 4s → ... up to 60s
- **Circuit breaker** - Opens after 5 consecutive failures, prevents retry storms
- **Fail-fast** - Permanent errors (401, 403) fail immediately without retry
- **Certificate validation** - Validates certificates at startup with actionable error messages

### Request Flow

```
1. Request arrives
         ↓
2. Check token cache
         ↓
3. Token expired? → OAuth2 token endpoint → Cache new token
         ↓
4. Add headers:
   - Authorization: Bearer <oauth_token>
   - X-Api-Key: <llm_api_key>  (if enterprise auth)
         ↓
5. Send to gateway (with CA cert / mTLS if configured)
         ↓
6. Parse OpenAI-compatible response
```

## Gateway Compatibility

The enterprise provider expects **OpenAI-compatible API format**:

**Chat endpoint:** `POST /chat/completions`
```json
{
  "model": "openai/gpt-4o",
  "messages": [{"role": "user", "content": "Hello"}]
}
```

Retrieval reranking runs locally via an INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base`) — see
[features/retrieval/reranking.md](../retrieval/reranking.md). It does
not consume the enterprise gateway. No `/embeddings` endpoint is ever
called either; fitz-sage dropped the embedding API entirely in v0.12.0.

Model strings are passed verbatim to the gateway. Common formats:
- `openai/gpt-4o` - Provider-prefixed (BMW gateway style)
- `gpt-4o` - Direct model name
- `my-deployment` - Azure deployment name

## Key Design Decisions

1. **No SDK dependencies** - Uses httpx directly. No OpenAI/Anthropic/Cohere SDKs.

2. **OpenAI-compatible API** - Works with any gateway that implements the OpenAI chat-completions format.

3. **Composite authentication** - M2M and API key auth can be combined for dual-header requirements.

4. **Fail-fast validation** - Certificates and environment variables are validated at startup.

5. **Transparent token refresh** - Application code doesn't need to handle token management.

## Files

- **Enterprise provider:** `fitz_sage/llm/providers/enterprise.py`
- **M2M authentication:** `fitz_sage/llm/auth/m2m.py`
- **Composite auth:** `fitz_sage/llm/auth/composite.py`
- **Config parser:** `fitz_sage/llm/config.py`
- **Certificate validation:** `fitz_sage/llm/auth/certificates.py`

## Example: Full Enterprise Setup

```yaml
# ~/.fitz/config/fitz_krag.yaml
synthesizer: enterprise/openai/gpt-4o
chat_base_url: https://llm.corp.internal/v1
collection: default

auth:
  type: enterprise
  token_url: https://auth.corp.internal/oauth/token
  client_id: ${CORP_CLIENT_ID}
  client_secret: ${CORP_CLIENT_SECRET}
  scope: llm.access
  llm_api_key_env: CORP_LLM_API_KEY
  llm_api_key_header: X-Api-Key
  cert_path: /etc/ssl/corp-ca-bundle.crt
  client_cert_path: /etc/ssl/client.crt
  client_key_path: /etc/ssl/client.key

# Storage: SQLite + FTS5 (auto-managed). No vector_db / pgvector knobs.
```

```bash
# Set environment variables
export CORP_CLIENT_ID="my-client-id"
export CORP_CLIENT_SECRET="my-client-secret"
export CORP_LLM_API_KEY="my-llm-api-key"

# Run Fitz
fitz query --source ./docs "What is the refund policy?"
```

## Benefits

| Direct Provider API | Enterprise Gateway |
|--------------------|-------------------|
| Requires internet access | Works behind corporate firewall |
| API key only | OAuth2 + API key + mTLS |
| Public CA certs | Custom internal PKI |
| Provider SDK required | No SDK dependencies |
| One auth method | Composite authentication |

## Related Features

- [**Configuration**](../CONFIG.md) - Full configuration reference
- [**Plugins**](../PLUGINS.md) - How providers are loaded
