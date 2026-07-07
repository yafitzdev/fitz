# fitz_sage/engines/fitz_krag/config/schema.py
"""
Configuration schema for Fitz KRAG (Knowledge Routing Augmented Generation) engine.

KRAG uses knowledge-type-aware access strategies instead of uniform chunk-based
retrieval. It stores raw files and symbol indexes, retrieves by address (pointer
to code symbol / document section), then reads content on demand.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from fitz_sage.config.defaults import DEFAULT_LOCAL_LLM_BASE_URL
from fitz_sage.core.config import BasePluginConfig


class FitzKragConfig(BasePluginConfig):
    """
    Fitz KRAG configuration.

    Minimal local config:
    ```yaml
    collection: my_project
    synthesizer: null
    ```

    Optional synthesis config (OpenAI):
    ```yaml
    synthesizer: openai/gpt-4o
    collection: my_project
    # OPENAI_API_KEY in env
    ```

    Note: fitz-sage uses no embedding model. Retrieval is BM25 + KRAG
    typed-unit routing (code symbols, sections, tables) + optional ONNX
    rerank. The ``retrieval intelligence stack`` does the semantic work that
    dense retrieval traditionally provides without requiring a chat model.
    """

    # ==========================================================================
    # Core Plugins (shared infrastructure)
    # ==========================================================================

    chat_fast: str | None = Field(
        default=None,
        description="Optional fast-tier chat model for synthesis and query intelligence",
    )

    chat_balanced: str | None = Field(
        default=None,
        description="Optional balanced-tier chat model for synthesis and query intelligence",
    )

    chat_smart: str | None = Field(
        default=None,
        description="Optional smart-tier chat model for synthesis and query intelligence",
    )

    # Per-role base URLs — used by the ``endpoint`` and ``enterprise``
    # provider names. Ignored for ``openai`` (which has a built-in
    # default URL) and ``azure_openai`` (which always requires its
    # own base_url at the spec level).
    chat_base_url: str | None = Field(
        default=DEFAULT_LOCAL_LLM_BASE_URL,
        description=(
            "HTTP endpoint for chat — used by the ``endpoint`` provider. "
            "Ignored by the managed ONNX enrichment provider."
        ),
    )

    vision_base_url: str | None = Field(
        default=None,
        description=(
            "HTTP endpoint for vision — used by the ``endpoint`` provider. "
            "If None, falls back to chat_base_url."
        ),
    )

    # Optional API key environment variable name when the endpoint
    # requires authentication (e.g. Together, Groq, Fireworks).
    chat_api_key_env: str | None = Field(
        default=None,
        description="Env var name for chat-endpoint API key (None = no auth).",
    )

    auth: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional auth block passed to chat providers. Use for endpoint API "
            "keys with custom headers, M2M OAuth2, or enterprise composite auth."
        ),
    )

    cert_path: str | None = Field(
        default=None,
        description="Optional CA certificate bundle path for enterprise/M2M auth.",
    )

    vision_api_key_env: str | None = Field(
        default=None,
        description="Env var name for vision-endpoint API key (None = no auth).",
    )

    rerank: str = Field(
        default="onnx",
        description=(
            "Reranker backend. 'onnx' (default) loads the INT8 ONNX "
            "cross-encoder (`Alibaba-NLP/gte-reranker-modernbert-base` "
            "by default; override with `onnx/<hf-model-id>`)."
        ),
    )

    vision: str | None = Field(
        default=None,
        description="Vision/VLM plugin for image description. None = disabled.",
    )

    parser: str = Field(
        default="cpu",
        description=(
            "Document parser. 'cpu' (default) — server-free, zero-model "
            "pypdfium2 PDF parsing on CPU. 'docling' / 'docling_vision' "
            "(figure description via the vision provider) / 'glm_ocr' — "
            "heavier opt-in parsers for scanned PDFs, figures, complex tables."
        ),
    )

    # ==========================================================================
    # Collection
    # ==========================================================================

    collection: str = Field(
        ...,
        description="Collection name (required)",
    )

    # ==========================================================================
    # Code Strategy
    # ==========================================================================

    code_search_mode: str = Field(
        default="auto",
        description=(
            "Code search mode: 'auto' = LLM structural search when chat "
            "is available with keyword+BM25 fallback, 'hybrid' = keyword+BM25 only"
        ),
    )

    code_languages: list[str] = Field(
        default=["python", "typescript", "java", "go"],
        description="Enabled code languages for ingestion",
    )

    summary_batch_size: int = Field(
        default=15,
        ge=1,
        description="Number of symbols per LLM summarization batch",
    )

    max_expansion_depth: int = Field(
        default=1,
        ge=0,
        description="Max depth for code context expansion (imports, class context)",
    )

    include_class_context: bool = Field(
        default=True,
        description="Include class signature + __init__ when expanding methods",
    )

    max_reference_expansions: int = Field(
        default=3,
        ge=0,
        description="Max same-file referenced symbols to include as context (0 = disabled)",
    )

    include_import_summaries: bool = Field(
        default=True,
        description="Include summaries of imported symbols as context",
    )

    max_import_expansions: int = Field(
        default=5,
        ge=0,
        description="Max imported symbol summaries to include as context",
    )

    # ==========================================================================
    # Retrieval
    # ==========================================================================

    top_addresses: int = Field(
        default=50,
        ge=1,
        description="Number of addresses to retrieve before reading",
    )

    top_read: int = Field(
        default=50,
        ge=1,
        description="Number of top addresses to read content for",
    )

    retrieval_workers: int = Field(
        default=4,
        ge=1,
        description=(
            "Max retrieval strategies run concurrently. Set to 1 to serialize "
            "LLM calls for single-model endpoint servers."
        ),
    )

    keyword_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for keyword (name) leg in code keyword+BM25 merge",
    )

    include_section_context: bool = Field(
        default=True,
        description="Include parent breadcrumb and child TOC for section addresses",
    )

    # ==========================================================================
    # Table Strategy
    # ==========================================================================

    table_extensions: list[str] = Field(
        default=[".csv", ".tsv"],
        description="File extensions to ingest as tables",
    )

    max_table_results: int = Field(
        default=100,
        ge=1,
        description="Max SQL result rows to include in context",
    )

    # ==========================================================================
    # Context Assembly
    # ==========================================================================

    max_context_tokens: int = Field(
        default=48000,
        ge=100,
        description="Max tokens in assembled context for LLM",
    )

    include_file_header: bool = Field(
        default=True,
        description="Include file path header in context blocks",
    )

    # ==========================================================================
    # Governance
    # ==========================================================================

    governance: str = Field(
        default="pyrrho",
        description=(
            "Epistemic governance classifier. 'pyrrho' (default) evaluates "
            "evidence prefixes with the local Pyrrho v2 classifier and maps "
            "SUFFICIENT / DISPUTED / INSUFFICIENT to the runtime modes; "
            "'pyrrho/<hf-model-id>' or 'pyrrho/<local-package-path>' swaps in "
            "a custom package."
        ),
    )

    # ==========================================================================
    # Generation
    # ==========================================================================

    synthesizer: str | None = Field(
        default=None,
        description=(
            "Optional chat provider/model spec for answer synthesis. "
            "None leaves retrieval/evidence as the default surface."
        ),
    )

    max_answer_tokens: int = Field(
        default=512,
        ge=1,
        description="Maximum tokens requested from the optional answer synthesizer.",
    )

    short_answer_tokens: int = Field(
        default=192,
        ge=1,
        description=(
            "Maximum tokens for specific factual synthesis questions. "
            "The effective cap is min(short_answer_tokens, max_answer_tokens)."
        ),
    )

    enable_citations: bool = Field(
        default=True,
        description="Enable [S1], [S2] citation markers in answers",
    )

    strict_grounding: bool = Field(
        default=True,
        description="Only generate answers from provided context",
    )

    # ==========================================================================
    # Query Intelligence
    # ==========================================================================

    query_intelligence: str | None = Field(
        default=None,
        description=(
            "Optional chat provider/model spec for LLM query prep. "
            "None uses the deterministic no-chat planner."
        ),
    )

    # ==========================================================================
    # Reranking
    # ==========================================================================

    rerank_k: int = Field(
        default=10,
        ge=1,
        description="Number of addresses to keep after reranking",
    )

    rerank_min_addresses: int = Field(
        default=2,
        ge=1,
        description="Minimum addresses before reranking is applied.",
    )

    # ==========================================================================
    # BM25 Code Search
    # ==========================================================================

    code_bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 search in code hybrid merge",
    )

    # ==========================================================================
    # Multi-Hop
    # ==========================================================================

    enable_multi_hop: bool = Field(
        default=True,
        description=(
            "Multi-hop iterative retrieval. Each hop's sufficiency is judged "
            "by the pyrrho governance classifier (no chat call), so a single "
            "hop is the common case."
        ),
    )

    max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum retrieval hops for multi-hop reasoning",
    )

    # ==========================================================================
    # Logging
    # ==========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
