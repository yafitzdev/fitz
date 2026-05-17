# fitz_stack/logging_tags.py
"""
Central place for defining logging subsystem tags.

These tags are used across fitz_sage modules to ensure
consistent, searchable log output.

Changing a tag here updates it project-wide.
"""

INGEST = "[INGEST]"
CHUNKING = "[CHUNKING]"
VALIDATION = "[VALIDATION]"
CHAT = "[CHAT]"
RETRIEVER = "[RETRIEVER]"
RERANK = "[RERANK]"
PROMPT = "[PROMPT]"
SOURCER = "[SOURCER]"
PIPELINE = "[PIPELINE]"
CLI = "[CLI]"
RGS = "[RGS]"  # retrieval-guided synthesis
STORAGE = "[STORAGE]"
GOVERNANCE = "[GOVERNANCE]"
