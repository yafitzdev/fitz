# fitz_sage/retrieval/detection/detectors/expansion.py
"""
Dictionary-based query term expansion.

Provides synonym and acronym terms for a query's words. These terms are
fused into the query-prep keyword set (``QueryBatcher``) so BM25 retrieval
sees vocabulary the user did not type — a deterministic, zero-LLM
complement to the LLM-generated semantic keywords.
"""

from __future__ import annotations

import re

_WORD_PATTERN = re.compile(r"\b\w+\b")

# Bidirectional synonyms - each term maps to its alternatives
SYNONYMS: dict[str, list[str]] = {
    # CRUD operations
    "delete": ["remove", "erase"],
    "remove": ["delete", "erase"],
    "create": ["add", "make", "generate"],
    "add": ["create", "insert"],
    "update": ["modify", "change", "edit"],
    "modify": ["update", "change", "edit"],
    "get": ["retrieve", "fetch", "obtain"],
    "retrieve": ["get", "fetch"],
    "fetch": ["get", "retrieve"],
    # Status/state
    "error": ["failure", "exception", "issue"],
    "failure": ["error", "exception"],
    "issue": ["problem", "error", "bug"],
    "bug": ["issue", "defect", "problem"],
    # Actions
    "start": ["begin", "launch", "initiate"],
    "stop": ["end", "halt", "terminate"],
    "run": ["execute", "perform"],
    "execute": ["run", "perform"],
    "install": ["setup", "deploy"],
    "setup": ["install", "configure"],
    "configure": ["setup", "set up"],
    # Common terms
    "file": ["document", "doc"],
    "document": ["file", "doc"],
    "folder": ["directory", "dir"],
    "directory": ["folder", "dir"],
    "user": ["account", "member"],
    "function": ["method", "procedure"],
    "method": ["function", "procedure"],
    "class": ["type", "object"],
    "list": ["array", "collection"],
    "array": ["list", "collection"],
    # Technical
    "api": ["endpoint", "interface"],
    "endpoint": ["api", "route"],
    "database": ["db", "datastore"],
    "db": ["database", "datastore"],
    "server": ["backend", "service"],
    "client": ["frontend", "app"],
    "request": ["call", "query"],
    "response": ["reply", "result"],
    # States
    "enable": ["activate", "turn on"],
    "disable": ["deactivate", "turn off"],
    "active": ["enabled", "on"],
    "inactive": ["disabled", "off"],
}

# Acronym expansions (one-way)
ACRONYMS: dict[str, str] = {
    "api": "application programming interface",
    "ui": "user interface",
    "ux": "user experience",
    "db": "database",
    "sql": "structured query language",
    "html": "hypertext markup language",
    "css": "cascading style sheets",
    "js": "javascript",
    "ts": "typescript",
    "url": "uniform resource locator",
    "http": "hypertext transfer protocol",
    "https": "hypertext transfer protocol secure",
    "json": "javascript object notation",
    "xml": "extensible markup language",
    "csv": "comma separated values",
    "pdf": "portable document format",
    "id": "identifier",
    "auth": "authentication",
    "config": "configuration",
    "env": "environment",
    "dev": "development",
    "prod": "production",
    "repo": "repository",
    "pr": "pull request",
    "ci": "continuous integration",
    "cd": "continuous deployment",
    "k8s": "kubernetes",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "vm": "virtual machine",
    "os": "operating system",
    "cpu": "central processing unit",
    "gpu": "graphics processing unit",
    "ram": "random access memory",
    "ssd": "solid state drive",
    "hdd": "hard disk drive",
    "iot": "internet of things",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "llm": "large language model",
    "rag": "retrieval augmented generation",
}


def expand_terms(query: str) -> list[str]:
    """Synonym + acronym expansion terms for a query's words.

    Returns extra keyword *terms* (not full-query variations) drawn from
    the fixed SYNONYMS / ACRONYMS dictionaries, deduplicated and in query
    order. Empty when the query has no expandable words.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for word in _WORD_PATTERN.findall(query.lower()):
        for syn in SYNONYMS.get(word, []):
            if syn not in seen:
                seen.add(syn)
                terms.append(syn)
        acronym = ACRONYMS.get(word)
        if acronym and acronym not in seen:
            seen.add(acronym)
            terms.append(acronym)
    return terms
