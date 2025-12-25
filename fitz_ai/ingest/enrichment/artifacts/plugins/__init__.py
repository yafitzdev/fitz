# fitz_ai/ingest/enrichment/artifacts/plugins/__init__.py
"""
Artifact generator plugins.

Each plugin in this directory is auto-discovered and registered.
Plugins must define:
    - plugin_name: str
    - plugin_type: str = "artifact"
    - supported_types: set[ContentType]
    - Generator class implementing ArtifactGenerator protocol
"""
