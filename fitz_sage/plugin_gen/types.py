# fitz_sage/plugin_gen/types.py
"""
Types and enums for the plugin generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PluginType(Enum):
    """Supported plugin types for generation."""

    # LLM providers (Python-based)
    LLM_CHAT = "llm-chat"
    LLM_EMBEDDING = "llm-embedding"
    LLM_RERANK = "llm-rerank"
    VECTOR_DB = "vector-db"
    RETRIEVAL = "retrieval"

    # Python-based plugins
    CHUNKER = "chunker"
    READER = "reader"

    @property
    def is_yaml(self) -> bool:
        """Check if this plugin type uses YAML format."""
        return self in {
            PluginType.LLM_CHAT,
            PluginType.LLM_EMBEDDING,
            PluginType.LLM_RERANK,
            PluginType.VECTOR_DB,
            PluginType.RETRIEVAL,
        }

    @property
    def is_python(self) -> bool:
        """Check if this plugin type uses Python format."""
        return not self.is_yaml

    @property
    def file_extension(self) -> str:
        """Get the file extension for this plugin type."""
        return ".yaml" if self.is_yaml else ".py"

    @property
    def display_name(self) -> str:
        """Human-readable name for display."""
        names = {
            PluginType.LLM_CHAT: "LLM Chat Provider",
            PluginType.LLM_EMBEDDING: "LLM Embedding Provider",
            PluginType.LLM_RERANK: "LLM Rerank Provider",
            PluginType.VECTOR_DB: "Vector Database",
            PluginType.RETRIEVAL: "Retrieval Strategy",
            PluginType.CHUNKER: "Document Chunker",
            PluginType.READER: "File Reader",
        }
        return names.get(self, self.value)

    @property
    def description(self) -> str:
        """Short description for CLI help."""
        descriptions = {
            PluginType.LLM_CHAT: "Connect to a chat LLM provider (e.g., OpenAI, Anthropic)",
            PluginType.LLM_EMBEDDING: "Connect to an embedding provider",
            PluginType.LLM_RERANK: "Connect to a reranking provider",
            PluginType.VECTOR_DB: "Connect to a vector database (e.g., Qdrant, Pinecone)",
            PluginType.RETRIEVAL: "Define a retrieval strategy",
            PluginType.CHUNKER: "Custom document chunking logic",
            PluginType.READER: "Custom file format reader",
        }
        return descriptions.get(self, "")

    def get_output_dir(self) -> Path:
        """Get the output directory for this plugin type in user home."""
        from fitz_sage.core.paths import FitzPaths

        # Use ~/.fitz/plugins/ (user home - same for everyone)
        base = FitzPaths.user_plugins()

        path_map = {
            PluginType.LLM_CHAT: base / "llm" / "chat",
            PluginType.LLM_EMBEDDING: base / "llm" / "embedding",
            PluginType.LLM_RERANK: base / "llm" / "rerank",
            PluginType.VECTOR_DB: base / "vector_db",
            PluginType.CHUNKER: base / "chunking",
            PluginType.READER: base / "reader",
            PluginType.RETRIEVAL: base / "retrieval",
        }

        return path_map.get(self, base / self.value)

    def get_example_plugins_dir(self) -> Path:
        """Get the directory with example plugins for this type."""
        import fitz_sage

        package_root = Path(fitz_sage.__file__).parent

        path_map = {
            PluginType.LLM_CHAT: package_root / "llm" / "chat",
            PluginType.LLM_EMBEDDING: package_root / "llm" / "embedding",
            PluginType.LLM_RERANK: package_root / "llm" / "rerank",
            PluginType.VECTOR_DB: package_root / "vector_db" / "plugins",
            PluginType.RETRIEVAL: package_root
            / "engines"
            / "fitz_krag"
            / "retrieval"
            / "strategies",
            PluginType.CHUNKER: package_root / "ingestion" / "chunking" / "plugins" / "default",
            PluginType.READER: package_root / "ingestion" / "reader" / "plugins",
        }

        return path_map.get(self, package_root)


class ReviewDecision(Enum):
    """User's decision after reviewing generated code."""

    APPROVE = "approve"  # Save the code as-is
    EDIT = "edit"  # User provided modified code
    REJECT = "reject"  # Don't save, abort generation


@dataclass
class ReviewResult:
    """Result of user review."""

    decision: ReviewDecision
    modified_code: Optional[str] = None  # Only set if decision is EDIT

    @classmethod
    def approve(cls) -> "ReviewResult":
        return cls(decision=ReviewDecision.APPROVE)

    @classmethod
    def edit(cls, new_code: str) -> "ReviewResult":
        return cls(decision=ReviewDecision.EDIT, modified_code=new_code)

    @classmethod
    def reject(cls) -> "ReviewResult":
        return cls(decision=ReviewDecision.REJECT)


class ValidationLevel(Enum):
    """Levels of validation for generated plugins."""

    SYNTAX = "syntax"  # YAML parse / Python AST parse
    SCHEMA = "schema"  # Pydantic schema / imports successful
    INTEGRATION = "integration"  # Loads through actual loader
    FUNCTIONAL = "functional"  # Runs on sample input (Python only)
    SEMANTIC = "semantic"  # LLM-as-judge (optional)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    level: ValidationLevel
    success: bool
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def passed(
        cls, level: ValidationLevel, details: Optional[Dict[str, Any]] = None
    ) -> "ValidationResult":
        """Create a passed validation result."""
        return cls(level=level, success=True, details=details or {})

    @classmethod
    def failed(cls, level: ValidationLevel, error: str, **details: Any) -> "ValidationResult":
        """Create a failed validation result."""
        return cls(level=level, success=False, error=error, details=details)


@dataclass
class GenerationResult:
    """Result of plugin generation."""

    success: bool
    plugin_type: PluginType
    path: Optional[Path] = None
    plugin_name: Optional[str] = None
    code: Optional[str] = None
    validations: List[ValidationResult] = field(default_factory=list)
    attempts: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def first_error(self) -> Optional[str]:
        """Get the first error message."""
        for v in self.validations:
            if not v.success and v.error:
                return v.error
        return self.errors[0] if self.errors else None


__all__ = [
    "PluginType",
    "ReviewDecision",
    "ReviewResult",
    "ValidationLevel",
    "ValidationResult",
    "GenerationResult",
]
