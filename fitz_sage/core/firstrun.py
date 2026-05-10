# fitz_sage/core/firstrun.py
"""
First-run experience for fitz-sage.

Auto-detects an OpenAI-compatible LLM endpoint and writes
``.fitz/config.yaml`` so the CLI can answer queries on first invocation.

Detection order:

1. **Local OpenAI-compatible HTTP server** — probes ports 8080, 8000,
   1234, 11434 for ``GET /v1/models``. The first responsive server
   wins; we read its ``/models`` listing to choose a chat model and
   (optionally) an embedding model. Recommended setup is llama.cpp's
   ``llama-server`` on port 8080.
2. **OpenAI cloud** — falls back to ``openai/gpt-4o-mini`` if
   ``OPENAI_API_KEY`` is set.
3. **No provider** — prints actionable setup instructions and aborts.

There is no Ollama-specific code path; Ollama users run it in
``/v1/`` mode and it's just another OpenAI-compatible server on
port 11434.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from fitz_sage.core.paths import FitzPaths

logger = logging.getLogger(__name__)

# Heuristics for distinguishing embedding models from chat models when
# scanning a server's ``/v1/models`` listing.
_EMBEDDING_PATTERNS = re.compile(
    r"embed|nomic|bge[\-_]|mxbai-embed|e5-|gte-|stella[\-_]",
    re.IGNORECASE,
)

# Common ports for OpenAI-compatible servers, ordered by preference:
# 8080 = llama-server, 8000 = vLLM/LM Studio, 1234 = LM Studio (older),
# 11434 = Ollama in /v1/ mode.
_PROBE_PORTS: tuple[int, ...] = (8080, 8000, 1234, 11434)
_PROBE_TIMEOUT_SECONDS = 0.5


@dataclass
class EndpointModel:
    """A model exposed by an OpenAI-compatible /v1/models listing."""

    id: str
    is_embedding: bool = False


@dataclass
class DetectedEndpoint:
    """A reachable OpenAI-compatible HTTP server."""

    base_url: str
    chat_models: list[EndpointModel] = field(default_factory=list)
    embedding_models: list[EndpointModel] = field(default_factory=list)


def needs_firstrun() -> bool:
    """Check if first-run setup is needed (no config exists)."""
    return not FitzPaths.config().exists()


def _classify(model_id: str) -> EndpointModel:
    """Classify a model id as chat vs embedding by name pattern."""
    return EndpointModel(
        id=model_id,
        is_embedding=bool(_EMBEDDING_PATTERNS.search(model_id)),
    )


def detect_endpoint() -> DetectedEndpoint | None:
    """
    Probe common local ports for an OpenAI-compatible server.

    Returns the first reachable endpoint with its ``/v1/models``
    listing, or ``None`` if nothing responds.
    """
    try:
        import httpx
    except ImportError:
        return None

    for port in _PROBE_PORTS:
        for host in ("localhost", "127.0.0.1"):
            base_url = f"http://{host}:{port}/v1"
            try:
                response = httpx.get(
                    f"{base_url}/models", timeout=_PROBE_TIMEOUT_SECONDS
                )
            except Exception:
                continue

            if response.status_code != 200:
                continue

            try:
                data = response.json()
            except Exception:
                continue

            raw_models = data.get("data") or data.get("models") or []
            classified: list[EndpointModel] = []
            for raw in raw_models:
                model_id = raw.get("id") or raw.get("name")
                if not model_id:
                    continue
                classified.append(_classify(str(model_id)))

            endpoint = DetectedEndpoint(base_url=base_url)
            for m in classified:
                if m.is_embedding:
                    endpoint.embedding_models.append(m)
                else:
                    endpoint.chat_models.append(m)
            return endpoint

    return None


def write_config(
    chat_fast: str,
    chat_balanced: str,
    chat_smart: str,
    embedding: str,
    *,
    chat_base_url: str | None = None,
    embedding_base_url: str | None = None,
    chat_api_key_env: str | None = None,
    embedding_api_key_env: str | None = None,
) -> Path:
    """Write ``.fitz/config.yaml`` from the resolved provider configuration."""
    config_path = FitzPaths.config()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Fitz Configuration",
        "# Docs: https://github.com/yafitzdev/fitz-sage/blob/main/docs/CONFIG.md",
        "",
        "# Chat models by tier (provider/model)",
        f"chat_fast: {chat_fast}",
        f"chat_balanced: {chat_balanced}",
        f"chat_smart: {chat_smart}",
        "",
        "# Embedding model",
        f"embedding: {embedding}",
        "",
        "# HTTP endpoints (used by the 'endpoint' provider)",
        f"chat_base_url: {chat_base_url if chat_base_url else 'null'}",
        f"embedding_base_url: {embedding_base_url if embedding_base_url else 'null'}",
        "vision_base_url: null",
        "",
        "# Optional API key environment variables",
        f"chat_api_key_env: {chat_api_key_env if chat_api_key_env else 'null'}",
        f"embedding_api_key_env: {embedding_api_key_env if embedding_api_key_env else 'null'}",
        "vision_api_key_env: null",
        "",
        "# Optional",
        "rerank: null",
        "vision: null",
        "",
        "collection: default",
        "",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _pick_chat_model(endpoint: DetectedEndpoint) -> str | None:
    """Choose a single chat model id from the listing.

    Heuristic: prefer the first non-embedding model. The user can
    rename later via the YAML.
    """
    if not endpoint.chat_models:
        return None
    return endpoint.chat_models[0].id


def _pick_embedding_model(endpoint: DetectedEndpoint) -> str | None:
    """Choose an embedding model id, or None if the server doesn't expose one."""
    if not endpoint.embedding_models:
        return None
    return endpoint.embedding_models[0].id


def _configure_from_endpoint(endpoint: DetectedEndpoint) -> bool:
    """Write a config bound to a detected local OpenAI-compatible server."""
    config_path = FitzPaths.config()

    chat_model = _pick_chat_model(endpoint)
    if chat_model is None:
        print(
            f"\n  Detected an OpenAI-compatible server at {endpoint.base_url} but "
            f"it lists no chat models. Load a chat model and run fitz again.\n"
        )
        return False

    embedding_model = _pick_embedding_model(endpoint)
    if embedding_model is None:
        # Many local servers run a single chat model on one process.
        # We default the embedding spec to a placeholder pointing at
        # the same URL — the user is expected to either start a second
        # server or (once chat-only retrieval lands) leave embedding
        # disabled.
        embedding_spec = "endpoint/nomic-embed-text-v1.5"
        embedding_url = "http://localhost:8081/v1"
        print(
            f"\n  Found chat model '{chat_model}' at {endpoint.base_url}; "
            f"no embedding model on the same server.\n"
            f"  Defaulting embedding to a separate llama-server on port 8081 "
            f"(start one or edit chat_only mode in config).\n"
        )
    else:
        embedding_spec = f"endpoint/{embedding_model}"
        embedding_url = endpoint.base_url

    chat_spec = f"endpoint/{chat_model}"
    write_config(
        chat_fast=chat_spec,
        chat_balanced=chat_spec,
        chat_smart=chat_spec,
        embedding=embedding_spec,
        chat_base_url=endpoint.base_url,
        embedding_base_url=embedding_url,
    )

    print(f"\n  Auto-configured from {endpoint.base_url}:")
    print(f"    chat:      {chat_model}")
    print(f"    embedding: {embedding_model or '(separate server, edit if needed)'}")
    print(f"\n  Config: {config_path}\n")
    return True


def _configure_from_openai_key() -> bool:
    """Write a config bound to OpenAI's public API."""
    config_path = FitzPaths.config()

    write_config(
        chat_fast="openai/gpt-4o-mini",
        chat_balanced="openai/gpt-4o-mini",
        chat_smart="openai/gpt-4o",
        embedding="openai/text-embedding-3-small",
        chat_base_url=None,
        embedding_base_url=None,
    )
    print("\n  Configured from OPENAI_API_KEY:")
    print("    chat (smart):    gpt-4o")
    print("    chat (fast/bal): gpt-4o-mini")
    print("    embedding:       text-embedding-3-small")
    print(f"\n  Config: {config_path}\n")
    return True


def _print_setup_instructions() -> None:
    """Print actionable setup instructions when no provider is reachable."""
    print("\n  No LLM provider found. Pick one of these:\n")
    print("  Option 1 — local llama.cpp (recommended):")
    print("    1. Install llama.cpp (https://github.com/ggerganov/llama.cpp)")
    print("    2. Download a chat model (.gguf) and start the server:")
    print("       llama-server -m chat-model.gguf --port 8080")
    print("    3. Optionally start an embedding server on port 8081:")
    print("       llama-server -m embed-model.gguf --port 8081 --embeddings")
    print("    4. Re-run `fitz query ...`\n")
    print("  Option 2 — OpenAI cloud:")
    print("    export OPENAI_API_KEY=sk-...")
    print("    Re-run `fitz query ...`\n")
    print("  Option 3 — any OpenAI-compatible cloud (Together, Groq, …):")
    print("    fitz query \"...\" --endpoint https://api.together.xyz/v1 \\")
    print("        --model meta-llama-3.1-70b --api-key-env TOGETHER_API_KEY\n")


def run_firstrun_setup() -> bool:
    """
    Interactive first-run setup.

    Returns ``True`` if a config was written successfully, ``False``
    otherwise (the caller exits with a non-zero code in that case).
    """
    # 1. Probe for a local OpenAI-compatible server.
    endpoint = detect_endpoint()
    if endpoint is not None:
        return _configure_from_endpoint(endpoint)

    # 2. Fall back to OpenAI cloud if a key is set.
    if os.getenv("OPENAI_API_KEY"):
        return _configure_from_openai_key()

    # 3. Nothing reachable — print actionable instructions.
    _print_setup_instructions()
    return False
