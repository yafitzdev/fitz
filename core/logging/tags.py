# fitz_stack/logging_tags.py
"""
Central place for defining logging subsystem tags.

These tags are used across fitz_ingest and fitz_rag modules to ensure
consistent, searchable log output.

Changing a tag here updates it project-wide.
"""

INGEST = "[INGEST]"
CHUNKING = "[CHUNKING]"
VECTOR_DB = "[VECTOR_DB]"
VALIDATION = "[VALIDATION]"
CHAT = "[CHAT]"
RETRIEVER = "[RETRIEVER]"
EMBEDDING = "[EMBEDDING]"
RERANK = "[RERANK]"
PROMPT = "[PROMPT]"
SOURCER = "[SOURCER]"
PIPELINE = "[PIPELINE]"
CLI = "[CLI]"
RGS = "[RGS]"     # retrieval-guided synthesis
VECTOR_SEARCH = "[VECTOR_SEARCH]"