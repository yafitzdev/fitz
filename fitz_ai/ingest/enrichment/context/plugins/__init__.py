# fitz_ai/ingest/enrichment/context/plugins/__init__.py
"""
Context builder plugins.

Each plugin in this directory is auto-discovered and registered.
Plugins must define:
    - plugin_name: str
    - plugin_type: str = "context"
    - supported_extensions: set[str]
    - Builder class implementing ContextBuilder protocol
"""
