# fitz_ai/llm/schema_defaults.py
"""
Schema defaults loader.

Reads default values from the master schema YAML files in fitz_ai/llm/schemas/.
This is the SINGLE SOURCE OF TRUTH for field defaults.

Usage:
    from fitz_ai.llm.schema_defaults import get_defaults, get_field_info

    # Get all defaults for chat plugins
    chat_defaults = get_defaults("chat")

    # Get info about a specific field
    field_info = get_field_info("chat", "auth.header_name")
    # Returns: {"required": False, "type": "string", "default": "Authorization", ...}
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# =============================================================================
# Constants
# =============================================================================

SCHEMAS_DIR = Path(__file__).parent / "schemas"

SCHEMA_FILES = {
    "chat": "chat_plugin_schema.yaml",
    "embedding": "embedding_plugin_schema.yaml",
    "rerank": "rerank_plugin_schema.yaml",
}


# =============================================================================
# Schema Loading
# =============================================================================


@lru_cache(maxsize=8)
def _load_schema(plugin_type: str) -> Dict[str, Any]:
    """Load and cache a schema file."""
    if plugin_type not in SCHEMA_FILES:
        raise ValueError(
            f"Unknown plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(SCHEMA_FILES.keys())}"
        )

    schema_path = SCHEMAS_DIR / SCHEMA_FILES[plugin_type]

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    with open(schema_path) as f:
        data = yaml.safe_load(f)

    return data.get("fields", {})


def get_field_info(plugin_type: str, field_name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific field.

    Args:
        plugin_type: "chat", "embedding", or "rerank"
        field_name: Field name (e.g., "auth.header_name")

    Returns:
        Field info dict with keys: required, type, default, options, description, example
        Returns None if field not found.
    """
    fields = _load_schema(plugin_type)
    return fields.get(field_name)


def get_all_fields(plugin_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Get all field definitions for a plugin type.

    Args:
        plugin_type: "chat", "embedding", or "rerank"

    Returns:
        Dict mapping field names to their definitions
    """
    return _load_schema(plugin_type)


def get_required_fields(plugin_type: str) -> List[str]:
    """
    Get list of required field names.

    Args:
        plugin_type: "chat", "embedding", or "rerank"

    Returns:
        List of field names where required=True
    """
    fields = _load_schema(plugin_type)
    return [name for name, info in fields.items() if info.get("required", False)]


def get_optional_fields(plugin_type: str) -> List[str]:
    """
    Get list of optional field names (those with defaults).

    Args:
        plugin_type: "chat", "embedding", or "rerank"

    Returns:
        List of field names where required=False
    """
    fields = _load_schema(plugin_type)
    return [name for name, info in fields.items() if not info.get("required", False)]


def get_defaults(plugin_type: str) -> Dict[str, Any]:
    """
    Get all default values for a plugin type.

    Returns a flat dict of field_name -> default_value.
    Only includes fields that have defaults defined.

    Args:
        plugin_type: "chat", "embedding", or "rerank"

    Returns:
        Dict mapping field names to their default values

    Example:
        >>> get_defaults("chat")
        {
            "version": "1.0",
            "auth.type": "bearer",
            "auth.header_name": "Authorization",
            "auth.header_format": "Bearer {key}",
            "endpoint.method": "POST",
            "endpoint.timeout": 120,
            ...
        }
    """
    fields = _load_schema(plugin_type)
    defaults = {}

    for name, info in fields.items():
        if "default" in info:
            defaults[name] = info["default"]

    return defaults


def get_nested_defaults(plugin_type: str) -> Dict[str, Any]:
    """
    Get defaults as a nested dict structure (matching YAML structure).

    Args:
        plugin_type: "chat", "embedding", or "rerank"

    Returns:
        Nested dict that can be merged with plugin YAML

    Example:
        >>> get_nested_defaults("chat")
        {
            "version": "1.0",
            "auth": {
                "type": "bearer",
                "header_name": "Authorization",
                "header_format": "Bearer {key}",
                "env_vars": []
            },
            "endpoint": {
                "method": "POST",
                "timeout": 120
            },
            ...
        }
    """
    flat_defaults = get_defaults(plugin_type)
    nested: Dict[str, Any] = {}

    # First pass: identify which parent fields have null defaults
    null_parents = set()
    for field_path, value in flat_defaults.items():
        if value is None and "." not in field_path:
            # This is a top-level field with null default (like health_check: null)
            null_parents.add(field_path)

    for field_path, value in flat_defaults.items():
        parts = field_path.split(".")

        # Skip nested fields if their parent is explicitly null
        # e.g., skip "health_check.path" if "health_check" default is null
        if len(parts) > 1 and parts[0] in null_parents:
            continue

        current = nested

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return nested


def get_field_options(plugin_type: str, field_name: str) -> Optional[List[Any]]:
    """
    Get allowed options for a field (if restricted).

    Args:
        plugin_type: "chat", "embedding", or "rerank"
        field_name: Field name

    Returns:
        List of allowed values, or None if not restricted
    """
    info = get_field_info(plugin_type, field_name)
    if info:
        return info.get("options")
    return None


def get_field_type(plugin_type: str, field_name: str) -> Optional[str]:
    """
    Get the type of a field.

    Args:
        plugin_type: "chat", "embedding", or "rerank"
        field_name: Field name

    Returns:
        Type string: "string", "integer", "float", "boolean", "list", "object"
    """
    info = get_field_info(plugin_type, field_name)
    if info:
        return info.get("type")
    return None


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_plugin_fields(
    plugin_type: str,
    plugin_data: Dict[str, Any],
    strict: bool = False,
) -> List[str]:
    """
    Validate plugin data against the schema.

    Args:
        plugin_type: "chat", "embedding", or "rerank"
        plugin_data: The plugin configuration dict
        strict: If True, warn about unknown fields

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    schema_fields = _load_schema(plugin_type)

    # Flatten plugin data for comparison
    flat_data = _flatten_dict(plugin_data)

    # Check required fields
    for field_name, field_info in schema_fields.items():
        if field_info.get("required", False):
            if field_name not in flat_data:
                errors.append(f"Missing required field: {field_name}")

    # Check field types and options
    for field_name, value in flat_data.items():
        if field_name not in schema_fields:
            if strict:
                errors.append(f"Unknown field: {field_name}")
            continue

        field_info = schema_fields[field_name]

        # Check options
        options = field_info.get("options")
        if options and value not in options:
            errors.append(
                f"Invalid value for {field_name}: {value!r}. Must be one of: {options}"
            )

    return errors


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dict with dot-separated keys."""
    items: Dict[str, Any] = {}

    for key, value in d.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            items.update(_flatten_dict(value, new_key))
        else:
            items[new_key] = value

    return items


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "get_field_info",
    "get_all_fields",
    "get_required_fields",
    "get_optional_fields",
    "get_defaults",
    "get_nested_defaults",
    "get_field_options",
    "get_field_type",
    "validate_plugin_fields",
]
