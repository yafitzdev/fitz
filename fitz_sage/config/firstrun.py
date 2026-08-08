# fitz_sage/config/firstrun.py
"""
First-run experience for fitz-sage.

Writes ``.fitz/config.yaml`` for user-configurable providers. Required
enrichment always uses the managed Qwen ONNX runtime and is not written as
configuration. If an OpenAI-compatible LLM endpoint is already available,
optional synthesis providers are configured too.

Detection order:

1. **Local OpenAI-compatible HTTP server** — optional synthesis only. Probes
   ports 8080, 8000, 1234, 11434 for ``GET /v1/models``. The first responsive
   server wins; we read its ``/models`` listing to choose a chat model.
2. **OpenAI cloud** — falls back to ``openai/gpt-4o-mini`` if
   ``OPENAI_API_KEY`` is set, again for optional synthesis only.
3. **No provider** — writes a minimal config. The first ingest downloads
   Qwen3 0.6B ONNX GenAI into the Hugging Face cache and runs it locally.

There is no Ollama-specific enrichment path; Ollama is only an optional
OpenAI-compatible endpoint for synthesis. fitz-sage uses no embeddings;
ingestion enrichment is mandatory.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from fitz_sage.config.defaults import DEFAULT_ENRICHMENT_MODEL, DEFAULT_LOCAL_LLM_BASE_URL
from fitz_sage.core.paths import FitzPaths

logger = logging.getLogger(__name__)


# Common ports for optional OpenAI-compatible synthesis servers, ordered by
# preference: 8080, 8000 = vLLM/LM Studio, 1234 = LM Studio (older), 11434 =
# Ollama in /v1/ mode.
_PROBE_PORTS: tuple[int, ...] = (8080, 8000, 1234, 11434)
_PROBE_TIMEOUT_SECONDS = 0.5


@dataclass
class EndpointModel:
    """A model exposed by an OpenAI-compatible /v1/models listing."""

    id: str


@dataclass
class DetectedEndpoint:
    """A reachable OpenAI-compatible HTTP server."""

    base_url: str
    chat_models: list[EndpointModel] = field(default_factory=list)


def needs_firstrun() -> bool:
    """Check if first-run setup is needed (no config exists)."""
    return not FitzPaths.config().exists()


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
                response = httpx.get(f"{base_url}/models", timeout=_PROBE_TIMEOUT_SECONDS)
            except Exception:
                continue

            if response.status_code != 200:
                continue

            try:
                data = response.json()
            except Exception:
                continue

            raw_models = data.get("data") or data.get("models") or []
            endpoint = DetectedEndpoint(base_url=base_url)
            for raw in raw_models:
                model_id = raw.get("id") or raw.get("name")
                if model_id:
                    endpoint.chat_models.append(EndpointModel(id=str(model_id)))
            return endpoint

    return None


def write_config(
    chat_fast: str,
    chat_balanced: str,
    chat_smart: str,
    *,
    chat_base_url: str | None = None,
    chat_api_key_env: str | None = None,
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
        f"synthesizer: {chat_smart}",
        "",
        "# HTTP endpoint (used only by endpoint/* providers)",
        f"chat_base_url: {chat_base_url if chat_base_url else DEFAULT_LOCAL_LLM_BASE_URL}",
        "vision_base_url: null",
        "",
        "# Optional API key environment variable",
        f"chat_api_key_env: {chat_api_key_env if chat_api_key_env else 'null'}",
        "vision_api_key_env: null",
        "",
        "# Retrieval defaults",
        "parser: cpu",
        "rerank: onnx",
        "# Pyrrho resolves its accepted default to an immutable model revision.",
        "governance: pyrrho",
        "vision: null",
        "",
        "collection: default",
        "",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def write_local_enrichment_config(
    *,
    chat_base_url: str = DEFAULT_LOCAL_LLM_BASE_URL,
) -> Path:
    """Write a default config; enrichment is managed internally."""
    config_path = FitzPaths.config()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Fitz Configuration",
        "# Docs: https://github.com/yafitzdev/fitz-sage/blob/main/docs/CONFIG.md",
        "",
        "# Query terms and optional background enrichment use managed ONNX Qwen.",
        "collection: default",
        "parser: cpu",
        "rerank: onnx",
        "# Pyrrho resolves its accepted default to an immutable model revision.",
        "governance: pyrrho",
        "# Optional endpoint providers use chat_base_url; managed Qwen ignores it.",
        f"chat_base_url: {chat_base_url}",
        "",
        "query_intelligence: null",
        "synthesizer: null",
        "",
    ]
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def _pick_chat_model(endpoint: DetectedEndpoint) -> str | None:
    """Choose a single chat model id from the listing.

    Heuristic: take the first model. The user can rename later via
    the YAML.
    """
    if not endpoint.chat_models:
        return None
    return endpoint.chat_models[0].id


def _configure_from_endpoint(endpoint: DetectedEndpoint) -> bool:
    """Write a config bound to a detected local OpenAI-compatible server."""
    config_path = FitzPaths.config()

    chat_model = _pick_chat_model(endpoint)
    if chat_model is None:
        config_path = write_local_enrichment_config(chat_base_url=endpoint.base_url)
        print(
            f"\n  Detected an OpenAI-compatible server at {endpoint.base_url}, "
            f"but it lists no chat models."
        )
        print(
            "  Wrote minimal config; semantic query terms and optional background "
            f"work use managed {DEFAULT_ENRICHMENT_MODEL}."
        )
        print(f"\n  Config: {config_path}\n")
        return True

    chat_spec = f"endpoint/{chat_model}"
    write_config(
        chat_fast=chat_spec,
        chat_balanced=chat_spec,
        chat_smart=chat_spec,
        chat_base_url=endpoint.base_url,
    )

    print(f"\n  Auto-configured from {endpoint.base_url}:")
    print(f"    chat:       {chat_model}")
    print(f"    query/background: {DEFAULT_ENRICHMENT_MODEL} (managed ONNX)")
    print(f"\n  Config: {config_path}\n")
    return True


def _configure_from_openai_key() -> bool:
    """Write a config bound to OpenAI's public API."""
    config_path = FitzPaths.config()

    write_config(
        chat_fast="openai/gpt-4o-mini",
        chat_balanced="openai/gpt-4o-mini",
        chat_smart="openai/gpt-4o",
        chat_base_url=DEFAULT_LOCAL_LLM_BASE_URL,
    )
    print("\n  Configured from OPENAI_API_KEY:")
    print("    chat (smart):    gpt-4o")
    print("    chat (fast/bal): gpt-4o-mini")
    print(f"    query/background: {DEFAULT_ENRICHMENT_MODEL} (managed ONNX)")
    print(f"\n  Config: {config_path}\n")
    return True


def _configure_local_enrichment_required() -> bool:
    """Write config when no optional chat endpoint is available."""
    config_path = write_local_enrichment_config()
    print("\n  No optional chat endpoint found.")
    print(
        "  Wrote minimal config; semantic query terms and optional background "
        f"work use managed {DEFAULT_ENRICHMENT_MODEL}."
    )
    print(
        "  The first model-backed query or enrichment operation downloads the "
        "managed Qwen3 0.6B ONNX GenAI weights locally."
    )
    print(f"\n  Config: {config_path}\n")
    return True


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

    # 3. Nothing reachable — write the required local runtime config and
    # tell the user how to satisfy it before ingestion.
    return _configure_local_enrichment_required()
