# fitz_sage/plugin_gen/context.py
"""
Context builder for LLM plugin generation.

Loads minimal plugin templates and schema definitions to build
prompts that help the LLM generate correct, working plugins.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fitz_sage.plugin_gen.test_inputs import get_expected_behavior
from fitz_sage.plugin_gen.types import PluginType

if TYPE_CHECKING:
    from fitz_sage.plugin_gen.library_context import LibraryContext

logger = logging.getLogger(__name__)

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"


# =============================================================================
# Template Loading
# =============================================================================


def load_template(plugin_type: PluginType) -> Optional[str]:
    """
    Load the minimal template for the given plugin type.

    Templates are stored at: templates/{plugin_type}/template.txt

    Args:
        plugin_type: Type of plugin to generate

    Returns:
        Template code or None if not found
    """
    template_path = TEMPLATES_DIR / plugin_type.name.lower() / "template.txt"

    if not template_path.exists():
        logger.warning(f"Template not found: {template_path}")
        return None

    try:
        return template_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read template {template_path}: {e}")
        return None


def load_example_plugin(plugin_type: PluginType) -> Optional[str]:
    """Load an example plugin for the given type (now loads templates)."""
    return load_template(plugin_type)


# =============================================================================
# Schema Information
# =============================================================================


def get_yaml_schema_info(plugin_type: PluginType) -> str:
    """Get schema requirements for YAML-based plugins (vector DB, retrieval)."""
    schemas = {
        PluginType.LLM_CHAT: """
Required fields:
- plugin_name: string (unique identifier)
- plugin_type: "chat"
- version: string (e.g., "1.0")
- provider:
    - name: string (provider identifier)
    - base_url: string (API base URL, must start with http:// or https://)
- auth:
    - type: "bearer" | "header" | "none"
    - env_vars: list of environment variable names for API key
    - header_name: string (default "Authorization")
    - header_format: string with {key} placeholder
- endpoint:
    - path: string (must start with /)
    - method: "POST" | "GET"
    - timeout: integer (seconds)
- defaults:
    - models:
        - smart: string (model for high-quality responses)
        - fast: string (model for speed)
        - balanced: string (cost-effective model)
    - temperature: float (0.0 - 1.0)
- request:
    - messages_transform: "openai_chat" | "anthropic_chat" | "cohere_chat" | "gemini_chat" | "ollama_chat"
    - param_map: dict mapping param names
- response:
    - content_path: JSONPath to extract response text (e.g., "choices[0].message.content")
    - metadata_paths: optional dict for token counts etc.
""",
        PluginType.LLM_EMBEDDING: """
Required fields:
- plugin_name: string (unique identifier)
- plugin_type: "embedding"
- version: string
- provider: same as chat
- auth: same as chat (MUST have env_vars with API key variable name)
- endpoint: same as chat
- defaults:
    - models:
        - smart: string (embedding model name)
        - fast: string
        - balanced: string
- request:
    - input_field: string (field name for input text)
    - input_wrap: "list" | "string" | "object"
- response:
    - embeddings_path: JSONPath to embedding vector (e.g., "data[0].embedding")
""",
        PluginType.LLM_RERANK: """
Required fields:
- plugin_name: string (unique identifier)
- plugin_type: "rerank"
- version: string
- provider: same as chat
- auth: same as chat (MUST have env_vars with API key variable name)
- endpoint: same as chat
- defaults:
    - models:
        - smart: string (rerank model name)
- request:
    - query_field: string (field name for query)
    - documents_field: string (field name for documents list)
- response:
    - results_path: JSONPath to results array
    - result_index_path: path to index within result
    - result_score_path: path to relevance score
""",
        PluginType.VECTOR_DB: """
Required fields:
- name: string (unique identifier)
- type: "vector_db"
- description: string (describe the vector database)
- features:
    - requires_uuid_ids: bool (true if DB requires UUID point IDs)
    - auto_detect: string|null (service name for auto-detection, e.g., "ollama")
    - supports_namespaces: bool (true if uses namespaces instead of collections)
- connection:
    - type: "http"
    - base_url: string with {{host}} and {{port}} placeholders
    - default_host: string (e.g., "localhost")
    - default_port: integer
    - auth:
        - type: "bearer" | "custom"
        - env_var: string (API key environment variable)
        - header: string (e.g., "Authorization")
        - scheme: string (e.g., "Bearer", or "" for no prefix)
        - optional: bool (true for self-hosted, false for cloud)
- operations (all use Jinja2 templates with {{collection}}, {{ids}}, etc.):
    - search: Find similar vectors (POST with query_vector, limit)
    - upsert: Insert/update points (POST/PUT with points array)
    - retrieve: Fetch points by IDs (REQUIRED for table storage)
    - count: Get collection size
    - create_collection: Create new collection
    - delete_collection: Delete collection
    - list_collections: List all collections
    - get_stats: Get collection statistics

Each operation needs:
- endpoint: string with {{collection}} placeholder
- method: "GET" | "POST" | "PUT" | "DELETE"
- body: request body template (for POST/PUT). Use "{{variable}}" for Jinja2 substitution.
- response:
    - results_path: DOT NOTATION path to results array (e.g., "result.points", NOT JSONPath like "$.result.points")
    - mapping: field names within each result item (e.g., id: "id", score: "score", payload: "payload")

CRITICAL SYNTAX RULES:
1. Use DOT NOTATION for paths, NOT JSONPath. Examples:
   - CORRECT: "result.points"
   - WRONG: "$.result.points"
2. Template variables use Jinja2 double braces: "{{variable}}"
3. Body values that are arrays should be quoted: '["value1", "value2"]'
4. The 'retrieve' operation is REQUIRED for table/CSV file support.
""",
        PluginType.RETRIEVAL: """
Required fields:
- name: string (strategy name)
- type: string (e.g., "dense", "hybrid")
- Any strategy-specific configuration
""",
    }

    return schemas.get(plugin_type, "Follow standard YAML plugin format.")


def get_python_protocol_info(plugin_type: PluginType) -> str:
    """Get protocol requirements for Python plugins."""
    protocols = {
        PluginType.CHUNKER: """
Required class structure:
```python
from dataclasses import dataclass, field
from typing import Any, Dict, List
from fitz_sage.core.chunk import Chunk

@dataclass
class MyChunker:
    plugin_name: str = field(default="my_chunker", repr=False)
    # Add configuration parameters here

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        '''
        Split text into chunks.

        Args:
            text: The text content to chunk
            base_meta: Metadata dict with doc_id, source_file, etc.

        Returns:
            List of Chunk objects
        '''
        # Implementation here
        pass
```

The Chunk class has these attributes:
- id: str (unique chunk ID, typically "{doc_id}:{chunk_index}")
- doc_id: str (document identifier)
- chunk_index: int (position in document)
- content: str (chunk text)
- metadata: Dict[str, Any] (inherited from base_meta)
""",
        PluginType.READER: """
Required class structure:
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

@dataclass
class MyReader:
    plugin_name: str = field(default="my_reader", repr=False)

    def ingest(self, path: Path) -> Iterable[RawDocument]:
        '''
        Read files and yield RawDocument objects.

        Args:
            path: Path to file or directory

        Yields:
            RawDocument objects with content and metadata
        '''
        pass
```

RawDocument has:
- content: str (file text content)
- metadata: Dict[str, Any] (source info)
""",
    }

    return protocols.get(plugin_type, "Implement required plugin protocol.")


# =============================================================================
# Prompt Building
# =============================================================================


def build_generation_prompt(
    plugin_type: PluginType,
    description: str,
    example_code: Optional[str] = None,
    library_context: Optional["LibraryContext"] = None,
) -> str:
    """
    Build the prompt for generating a plugin.

    Args:
        plugin_type: Type of plugin to generate
        description: User's description of what they want
        example_code: Optional example plugin code
        library_context: Optional context about an external library to use

    Returns:
        Complete prompt for the LLM
    """
    if example_code is None:
        example_code = load_example_plugin(plugin_type)

    # Build prompt parts
    parts = [
        f"Generate a {plugin_type.display_name} plugin for: {description}",
        "",
        "## Requirements",
        "",
    ]

    # Add schema or protocol info
    if plugin_type.is_yaml:
        parts.append(get_yaml_schema_info(plugin_type))
    else:
        parts.append(get_python_protocol_info(plugin_type))

    # Add expected behavior
    parts.extend(
        [
            "",
            "## Expected Behavior",
            "",
            get_expected_behavior(plugin_type),
        ]
    )

    # Add library context if provided
    if library_context:
        parts.extend(
            [
                "",
                "## External Library to Use",
                "",
                f"The user explicitly requested using the `{library_context.name}` library.",
                f"Install command: `{library_context.install_command}`",
                "",
                f"**Summary**: {library_context.summary}",
                "",
                "### Library Documentation",
                "",
                library_context.readme_excerpt,
                "",
                "**IMPORTANT**: You MUST use this library in your implementation. "
                "Import it at the top of the file and use its API as shown in the documentation above.",
            ]
        )

    # Add example if available
    if example_code:
        parts.extend(
            [
                "",
                "## Example Plugin Structure (for reference)",
                "",
                f"```{'yaml' if plugin_type.is_yaml else 'python'}",
                example_code.strip(),
                "```",
            ]
        )
        if library_context:
            parts.append(
                "\nNote: The example above shows the plugin structure. "
                f"You should adapt it to use the `{library_context.name}` library."
            )

    # Add instructions - vary based on whether external library is used
    extension = plugin_type.file_extension

    if library_context:
        # Instructions when using an external library
        parts.extend(
            [
                "",
                "## Instructions",
                "",
                f"Generate a complete, working {extension} plugin using the "
                f"`{library_context.name}` library.",
                "- Output ONLY the plugin code, no explanations",
                "- Do NOT include file path comments at the top (no '# fitz_sage/...' lines)",
                "- Ensure all required fields are present",
                f"- Import and use `{library_context.name}` - this is required",
                "- Follow the library's documented API patterns",
                "- Handle the library's output format correctly",
                "",
                f"Output the complete {extension} code:",
            ]
        )
    else:
        # Standard instructions (no external library)
        parts.extend(
            [
                "",
                "## Instructions",
                "",
                f"Generate a complete, working {extension} plugin.",
                "- Output ONLY the plugin code, no explanations",
                "- Do NOT include file path comments at the top (no '# fitz_sage/...' lines)",
                "- Ensure all required fields are present",
                "- Use realistic values based on the description",
                "- The plugin should work immediately after setting the required API key",
                "- IMPORTANT: Use only Python standard library or packages already imported in fitz-sage",
                "- Do NOT use nltk, spacy, or other NLP libraries that require extra downloads",
                "- For sentence splitting, use regex: re.split(r'(?<=[.!?])\\s+', text)",
                "",
                f"Output the complete {extension} code:",
            ]
        )

    return "\n".join(parts)


def build_retry_prompt(
    plugin_type: PluginType,
    description: str,
    previous_code: str,
    error_feedback: str,
    library_context: Optional["LibraryContext"] = None,
) -> str:
    """
    Build a retry prompt after validation failure.

    Args:
        plugin_type: Type of plugin
        description: Original user description
        previous_code: The code that failed validation
        error_feedback: Formatted validation error
        library_context: Optional library context to preserve across retries

    Returns:
        Prompt for retry attempt
    """
    parts = [
        f"Your previous {plugin_type.display_name} plugin for '{description}' failed validation.",
        "",
        "## Error",
        "",
        error_feedback,
        "",
        "## Your Previous Code",
        "",
        f"```{'yaml' if plugin_type.is_yaml else 'python'}",
        previous_code.strip(),
        "```",
    ]

    # Include library context on retry so LLM doesn't forget about it
    if library_context:
        parts.extend(
            [
                "",
                "## Required Library",
                "",
                f"You MUST use the `{library_context.name}` library in your fix.",
                f"Install: `{library_context.install_command}`",
                "",
                "Key API patterns:",
                "",
                library_context.readme_excerpt,
            ]
        )

    parts.extend(
        [
            "",
            "## Instructions",
            "",
            "Fix the error and regenerate the complete plugin.",
            "- Output ONLY the corrected plugin code",
            "- Ensure all required fields are present",
            "- Do not remove any necessary fields",
        ]
    )

    if library_context:
        parts.append(f"- Continue using the `{library_context.name}` library")

    parts.extend(
        [
            "",
            "Output the corrected code:",
        ]
    )

    return "\n".join(parts)


def extract_code_from_response(response: str, plugin_type: PluginType) -> str:
    """
    Extract code from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response
        plugin_type: Type of plugin (for extension)

    Returns:
        Extracted code string
    """
    response = response.strip()

    # Try to extract from markdown code blocks
    lang = "yaml" if plugin_type.is_yaml else "python"

    # Check for ```yaml or ```python blocks
    for marker in [f"```{lang}", "```"]:
        if marker in response:
            start = response.find(marker) + len(marker)
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()

    # If no code block, return as-is (might be raw code)
    return response


__all__ = [
    "load_template",
    "load_example_plugin",
    "get_yaml_schema_info",
    "get_python_protocol_info",
    "build_generation_prompt",
    "build_retry_prompt",
    "extract_code_from_response",
    "TEMPLATES_DIR",
]
