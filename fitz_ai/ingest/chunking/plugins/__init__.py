# fitz_ai/ingest/chunking/plugins/__init__.py
"""
Plugin namespace for chunker plugins.
Modules inside this package will be auto-discovered.

This __init__.py file can also be used to manually register plugins
if auto-discovery fails.
"""

# Optional: Manual registration fallback
# Uncomment this section if auto-discovery isn't working:

# from fitz_ai.core.registry import CHUNKING_REGISTRY
# 
# # Import and register each plugin explicitly
# try:
#     from .simple import SimpleChunker
#     CHUNKING_REGISTRY.register(SimpleChunker)
# except Exception:
#     pass
# 
# try:
#     from .pdf_sections import PdfSectionChunker
#     CHUNKING_REGISTRY.register(PdfSectionChunker)
# except Exception:
#     pass