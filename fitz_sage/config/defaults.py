# fitz_sage/config/defaults.py
"""Shared product defaults used by config loading and first-run setup."""

DEFAULT_LOCAL_LLM_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_ENRICHMENT_MODEL = "qwen3.5-0.8b"

__all__ = [
    "DEFAULT_LOCAL_LLM_BASE_URL",
    "DEFAULT_ENRICHMENT_MODEL",
]
