# fitz_sage/engines/fitz_krag/config/schema.py
"""
Configuration schema for Fitz KRAG (Knowledge Routing Augmented Generation) engine.

KRAG uses knowledge-type-aware access strategies instead of uniform chunk-based
retrieval. It stores raw files and symbol indexes, retrieves by address (pointer
to code symbol / document section), then reads content on demand.
"""

from __future__ import annotations

from pydantic import Field

from fitz_sage.core.config import BasePluginConfig


class FitzKragConfig(BasePluginConfig):
    """
    Fitz KRAG configuration.

    Minimal local config (default — assumes one llama-server with
    a chat model on localhost:8080):
    ```yaml
    chat_fast: endpoint/qwen2.5-7b-instruct
    chat_balanced: endpoint/qwen2.5-7b-instruct
    chat_smart: endpoint/qwen2.5-7b-instruct
    chat_base_url: http://localhost:8080/v1
    collection: my_project
    ```

    Cloud config (OpenAI):
    ```yaml
    chat_smart: openai/gpt-4o
    chat_balanced: openai/gpt-4o-mini
    chat_fast: openai/gpt-4o-mini
    collection: my_project
    # OPENAI_API_KEY in env
    ```

    Note: fitz-sage uses no embedding model. Retrieval is BM25 + KRAG
    typed-unit routing (code symbols, sections, tables) + LLM rerank.
    The ``retrieval intelligence stack`` does the semantic work that
    dense retrieval traditionally provides — without the failure mode
    of surface-similar-but-wrong dense candidates.
    """

    # ==========================================================================
    # Core Plugins (shared infrastructure)
    # ==========================================================================

    chat_fast: str = Field(
        default="endpoint/qwen2.5-7b-instruct",
        description="Chat model for detection and query analysis (provider/model)",
    )

    chat_balanced: str = Field(
        default="endpoint/qwen2.5-7b-instruct",
        description="Chat model for general queries (provider/model)",
    )

    chat_smart: str = Field(
        default="endpoint/qwen2.5-7b-instruct",
        description="Chat model for complex generation (provider/model)",
    )

    # Per-role base URLs — used by the ``endpoint`` and ``enterprise``
    # provider names. Ignored for ``openai`` (which has a built-in
    # default URL) and ``azure_openai`` (which always requires its
    # own base_url at the spec level).
    chat_base_url: str | None = Field(
        default="http://localhost:8080/v1",
        description=(
            "HTTP endpoint for chat — used by the ``endpoint`` provider. "
            "Default is a local llama-server on port 8080."
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

    vision_api_key_env: str | None = Field(
        default=None,
        description="Env var name for vision-endpoint API key (None = no auth).",
    )

    rerank: str | None = Field(
        default="onnx",
        description=(
            "Reranker backend. 'onnx' (default) loads the INT8 ONNX "
            "cross-encoder (`Alibaba-NLP/gte-reranker-modernbert-base` "
            "by default; override with `onnx/<hf-model-id>`). None "
            "disables reranking entirely."
        ),
    )

    vision: str | None = Field(
        default=None,
        description="Vision/VLM plugin for image description. None = disabled.",
    )

    parser: str = Field(
        default="docling",
        description=(
            "Document parser: 'docling', 'docling_vision' (uses vision "
            "provider for figure description), or 'glm_ocr' (hybrid "
            "pypdfium2 + GLM-OCR via the configured vision endpoint)."
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
            "available with hybrid fallback, 'hybrid' = BM25 + semantic only"
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
            "LLM calls for single-model local servers (LM Studio, llama-server)."
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

    governance: str | None = Field(
        default="pyrrho",
        description=(
            "Epistemic governance classifier. 'pyrrho' (default) labels each "
            "answer TRUSTWORTHY / DISPUTED / ABSTAIN via an INT8 ONNX "
            "classifier; 'pyrrho/<hf-model-id>' swaps in a custom fine-tune. "
            "null disables governance entirely."
        ),
    )

    # ==========================================================================
    # Generation
    # ==========================================================================

    enable_citations: bool = Field(
        default=True,
        description="Enable [S1], [S2] citation markers in answers",
    )

    strict_grounding: bool = Field(
        default=True,
        description="Only generate answers from provided context",
    )

    # ==========================================================================
    # Detection
    # ==========================================================================

    enable_detection: bool = Field(
        default=True,
        description="Enable shared detection (temporal, comparison, expansion awareness)",
    )

    # ==========================================================================
    # Query Intelligence
    # ==========================================================================

    enable_query_rewriting: bool = Field(
        default=True,
        description="Enable LLM-based query rewriting for retrieval optimization",
    )

    enable_multi_query: bool = Field(
        default=True,
        description="Enable multi-query expansion for long/complex queries",
    )

    multi_query_min_length: int = Field(
        default=300,
        ge=50,
        description="Minimum query character length to trigger multi-query expansion",
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
        default=20,
        ge=1,
        description="Minimum addresses before reranking is applied (skip if fewer)",
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
    # Enrichment
    # ==========================================================================

    enable_enrichment: bool = Field(
        default=True,
        description="Enable keyword/entity extraction during ingestion",
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
    # Hierarchy
    # ==========================================================================

    enable_hierarchy: bool = Field(
        default=True,
        description="Enable L1/L2 hierarchical summaries during ingestion",
    )

    # ==========================================================================
    # Logging
    # ==========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
