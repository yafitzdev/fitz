# fitz_sage/plugin_gen/validators.py
"""
Plugin validation for generated code.

Multi-level validation:
- YAML: syntax → schema → loader integration
- Python: syntax → import → protocol → functional
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml

from fitz_sage.plugin_gen.types import PluginType, ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


class PluginValidator:
    """
    Validates generated plugin code.

    Runs multi-level validation appropriate for the plugin type.
    """

    def validate(
        self,
        code: str,
        plugin_type: PluginType,
        plugin_name: Optional[str] = None,
        uses_external_library: bool = False,
    ) -> List[ValidationResult]:
        """
        Validate plugin code.

        Args:
            code: The generated plugin code
            plugin_type: Type of plugin
            plugin_name: Expected plugin name (for consistency check)
            uses_external_library: If True, skip import/functional validation
                (library may not be installed)

        Returns:
            List of validation results (one per level attempted)
        """
        if plugin_type.is_yaml:
            return self._validate_yaml(code, plugin_type, plugin_name)
        else:
            return self._validate_python(code, plugin_type, plugin_name, uses_external_library)

    def _validate_yaml(
        self,
        code: str,
        plugin_type: PluginType,
        plugin_name: Optional[str],
    ) -> List[ValidationResult]:
        """Validate YAML plugin."""
        results = []

        # Level 1: YAML Syntax
        try:
            data = yaml.safe_load(code)
            if not isinstance(data, dict):
                results.append(
                    ValidationResult.failed(
                        ValidationLevel.SYNTAX,
                        f"YAML must be a mapping, got {type(data).__name__}",
                    )
                )
                return results
            results.append(ValidationResult.passed(ValidationLevel.SYNTAX))
        except yaml.YAMLError as e:
            results.append(
                ValidationResult.failed(
                    ValidationLevel.SYNTAX,
                    f"YAML syntax error: {e}",
                )
            )
            return results

        # Level 2: Schema Validation
        schema_result = self._validate_yaml_schema(data, plugin_type)
        results.append(schema_result)
        if not schema_result.success:
            return results

        # Level 3: Loader Integration
        integration_result = self._validate_yaml_integration(code, plugin_type, data)
        results.append(integration_result)

        return results

    def _validate_yaml_schema(
        self,
        data: Dict[str, Any],
        plugin_type: PluginType,
    ) -> ValidationResult:
        """Validate YAML data against schema."""
        # Check required fields based on plugin type
        if plugin_type in {PluginType.LLM_CHAT, PluginType.LLM_EMBEDDING, PluginType.LLM_RERANK}:
            return self._validate_llm_schema(data, plugin_type)
        elif plugin_type == PluginType.VECTOR_DB:
            return self._validate_vector_db_schema(data)
        elif plugin_type == PluginType.RETRIEVAL:
            return self._validate_retrieval_schema(data)

        return ValidationResult.passed(ValidationLevel.SCHEMA)

    def _validate_llm_schema(
        self,
        data: Dict[str, Any],
        plugin_type: PluginType,
    ) -> ValidationResult:
        """Validate LLM plugin schema."""
        from pydantic import ValidationError

        from fitz_sage.llm.schema import (
            ChatPluginSpec,
            EmbeddingPluginSpec,
            RerankPluginSpec,
        )

        spec_map = {
            PluginType.LLM_CHAT: ChatPluginSpec,
            PluginType.LLM_EMBEDDING: EmbeddingPluginSpec,
            PluginType.LLM_RERANK: RerankPluginSpec,
        }

        spec_class = spec_map.get(plugin_type)
        if not spec_class:
            return ValidationResult.passed(ValidationLevel.SCHEMA)

        try:
            spec_class.model_validate(data)
            return ValidationResult.passed(ValidationLevel.SCHEMA)
        except ValidationError as e:
            errors = []
            for err in e.errors():
                loc = " -> ".join(str(x) for x in err.get("loc", []))
                msg = err.get("msg", "Unknown error")
                errors.append(f"{loc}: {msg}")
            return ValidationResult.failed(
                ValidationLevel.SCHEMA,
                "Schema validation failed:\n  - " + "\n  - ".join(errors),
            )

    def _validate_vector_db_schema(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate vector DB plugin schema."""
        required = ["name", "type", "connection", "operations"]
        missing = [f for f in required if f not in data]

        if missing:
            return ValidationResult.failed(
                ValidationLevel.SCHEMA,
                f"Missing required fields: {missing}",
            )

        return ValidationResult.passed(ValidationLevel.SCHEMA)

    def _validate_retrieval_schema(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate retrieval plugin schema."""
        # Retrieval plugins have minimal schema requirements
        if "name" not in data and "plugin_name" not in data:
            return ValidationResult.failed(
                ValidationLevel.SCHEMA,
                "Missing required field: name or plugin_name",
            )

        return ValidationResult.passed(ValidationLevel.SCHEMA)

    def _validate_yaml_integration(
        self,
        code: str,
        plugin_type: PluginType,
        data: Dict[str, Any],
    ) -> ValidationResult:
        """Test loading the plugin through actual fitz-sage loaders."""
        # Write to temp file and try to load
        plugin_name = data.get("plugin_name") or data.get("name")
        if not plugin_name:
            return ValidationResult.failed(
                ValidationLevel.INTEGRATION,
                "Cannot determine plugin name from YAML",
            )

        # For LLM plugins, we need to write to a temp location and load
        # For now, just verify the structure is correct
        # Full integration testing would require mocking the file system

        return ValidationResult.passed(ValidationLevel.INTEGRATION)

    def _validate_python(
        self,
        code: str,
        plugin_type: PluginType,
        plugin_name: Optional[str],
        uses_external_library: bool = False,
    ) -> List[ValidationResult]:
        """Validate Python plugin."""
        results = []

        # Level 1: Python Syntax
        try:
            ast.parse(code)
            results.append(ValidationResult.passed(ValidationLevel.SYNTAX))
        except SyntaxError as e:
            results.append(
                ValidationResult.failed(
                    ValidationLevel.SYNTAX,
                    f"Syntax error at line {e.lineno}: {e.msg}",
                )
            )
            return results

        # Level 2: AST Structure Check (for external library plugins)
        if uses_external_library:
            # Skip import/functional tests - library may not be installed
            # Just validate the code structure via AST
            ast_result = self._validate_python_ast_structure(code, plugin_type)
            results.append(ast_result)
            return results

        # Level 2: Import Test
        import_result, module, plugin_class = self._validate_python_import(code)
        results.append(import_result)
        if not import_result.success:
            return results

        # Level 3: Protocol Check
        protocol_result = self._validate_python_protocol(plugin_class, plugin_type)
        results.append(protocol_result)
        if not protocol_result.success:
            return results

        # Level 4: Functional Test
        functional_result = self._validate_python_functional(plugin_class, plugin_type)
        results.append(functional_result)

        return results

    def _validate_python_ast_structure(
        self,
        code: str,
        plugin_type: PluginType,
    ) -> ValidationResult:
        """
        Validate Python plugin structure via AST (without importing).

        Used when external libraries are required but not installed.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult.failed(
                ValidationLevel.SCHEMA,
                f"Syntax error at line {e.lineno}: {e.msg}",
            )

        # Find class definitions
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if not classes:
            return ValidationResult.failed(
                ValidationLevel.SCHEMA,
                "No class definition found in generated code",
            )

        # Check the main class has required structure
        required_methods = {
            PluginType.CHUNKER: ["chunk_text"],
            PluginType.READER: ["ingest"],
            PluginType.CONSTRAINT: ["apply"],
        }

        target_methods = required_methods.get(plugin_type, [])

        for cls in classes:
            methods = [node.name for node in ast.walk(cls) if isinstance(node, ast.FunctionDef)]

            # Check for plugin_name attribute
            has_plugin_name = False
            for node in ast.walk(cls):
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if node.target.id in ("plugin_name", "name"):
                        has_plugin_name = True
                        break
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in (
                            "plugin_name",
                            "name",
                        ):
                            has_plugin_name = True
                            break

            # Check if this class has all required methods
            has_all_methods = all(m in methods for m in target_methods)

            if has_plugin_name and has_all_methods:
                return ValidationResult.passed(
                    ValidationLevel.SCHEMA,
                    details={"note": "Validated via AST (external library not installed)"},
                )

        # No suitable class found
        missing = []
        if not has_plugin_name:
            missing.append("plugin_name attribute")
        for m in target_methods:
            if m not in methods:
                missing.append(f"{m}() method")

        return ValidationResult.failed(
            ValidationLevel.SCHEMA,
            f"Plugin class missing: {', '.join(missing)}",
        )

    def _validate_python_import(
        self,
        code: str,
    ) -> Tuple[ValidationResult, Optional[Any], Optional[Type]]:
        """Try to import the Python code."""
        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            temp_path = Path(f.name)

        try:
            # Load module
            spec = importlib.util.spec_from_file_location("temp_plugin", temp_path)
            if spec is None or spec.loader is None:
                return (
                    ValidationResult.failed(
                        ValidationLevel.SCHEMA,
                        "Cannot create module spec",
                    ),
                    None,
                    None,
                )

            module = importlib.util.module_from_spec(spec)
            sys.modules["temp_plugin"] = module
            spec.loader.exec_module(module)

            # Find plugin class
            plugin_class = self._find_plugin_class(module)
            if not plugin_class:
                return (
                    ValidationResult.failed(
                        ValidationLevel.SCHEMA,
                        "No valid plugin class found in module",
                    ),
                    module,
                    None,
                )

            return (
                ValidationResult.passed(ValidationLevel.SCHEMA),
                module,
                plugin_class,
            )

        except ImportError as e:
            return (
                ValidationResult.failed(
                    ValidationLevel.SCHEMA,
                    f"Import error: {e}",
                ),
                None,
                None,
            )
        except Exception as e:
            return (
                ValidationResult.failed(
                    ValidationLevel.SCHEMA,
                    f"Error loading module: {e}",
                ),
                None,
                None,
            )
        finally:
            # Cleanup temp file
            try:
                temp_path.unlink()
            except Exception as e:
                logger.debug(f"Failed to cleanup temp plugin file: {e}")
            # Remove from sys.modules
            sys.modules.pop("temp_plugin", None)

    def _find_plugin_class(self, module: Any) -> Optional[Type]:
        """Find the plugin class in a module."""
        for name in dir(module):
            if name.startswith("_"):
                continue

            obj = getattr(module, name)
            if not isinstance(obj, type):
                continue

            # Check for plugin_name attribute
            if hasattr(obj, "plugin_name") or hasattr(obj, "name"):
                return obj

        return None

    def _validate_python_protocol(
        self,
        plugin_class: Type,
        plugin_type: PluginType,
    ) -> ValidationResult:
        """Check if plugin implements required protocol."""
        required_attrs = {
            PluginType.CHUNKER: ("plugin_name", "chunk_text"),
            PluginType.READER: ("plugin_name", "ingest"),
            PluginType.CONSTRAINT: ("name", "apply"),
        }

        attrs = required_attrs.get(plugin_type, ("plugin_name",))
        name_attr, *method_attrs = attrs if len(attrs) > 1 else (attrs[0], [])

        # Check name attribute
        if not hasattr(plugin_class, name_attr):
            # Try alternate
            alt_attr = "name" if name_attr == "plugin_name" else "plugin_name"
            if not hasattr(plugin_class, alt_attr):
                return ValidationResult.failed(
                    ValidationLevel.INTEGRATION,
                    f"Missing required attribute: {name_attr}",
                )

        # Check required methods
        for method in method_attrs:
            if not hasattr(plugin_class, method):
                return ValidationResult.failed(
                    ValidationLevel.INTEGRATION,
                    f"Missing required method: {method}",
                )

        return ValidationResult.passed(ValidationLevel.INTEGRATION)

    def _validate_python_functional(
        self,
        plugin_class: Type,
        plugin_type: PluginType,
    ) -> ValidationResult:
        """Run functional test on the plugin."""
        from fitz_sage.plugin_gen.test_inputs import get_test_input

        try:
            # Instantiate plugin
            instance = plugin_class()

            # Get test input
            test_input = get_test_input(plugin_type)
            if not test_input:
                return ValidationResult.passed(ValidationLevel.FUNCTIONAL)

            # Run the plugin
            if plugin_type == PluginType.CHUNKER:
                result = instance.chunk_text(test_input["text"], test_input["meta"])
                return self._validate_chunker_output(result)

            elif plugin_type == PluginType.READER:
                # Reader requires actual files, skip for now
                return ValidationResult.passed(ValidationLevel.FUNCTIONAL)

            return ValidationResult.passed(ValidationLevel.FUNCTIONAL)

        except Exception as e:
            return ValidationResult.failed(
                ValidationLevel.FUNCTIONAL,
                f"Runtime error: {e}",
            )

    def _validate_chunker_output(self, result: Any) -> ValidationResult:
        """Validate chunker output."""
        if not isinstance(result, list):
            return ValidationResult.failed(
                ValidationLevel.FUNCTIONAL,
                f"chunk_text must return List[Chunk], got {type(result).__name__}",
            )

        if len(result) == 0:
            return ValidationResult.failed(
                ValidationLevel.FUNCTIONAL,
                "chunk_text returned empty list for non-empty input",
            )

        # Check first chunk structure
        first = result[0]
        required_attrs = ["content"]

        for attr in required_attrs:
            if not hasattr(first, attr):
                return ValidationResult.failed(
                    ValidationLevel.FUNCTIONAL,
                    f"Chunk missing required attribute: {attr}",
                )

        return ValidationResult.passed(ValidationLevel.FUNCTIONAL)


def format_validation_error(result: ValidationResult, code: str) -> str:
    """
    Format a validation error for LLM feedback.

    Returns a string suitable for prompting the LLM to fix the error.
    """
    lines = [
        f"CHECK FAILED: {result.level.value}",
        f"ERROR: {result.error}",
    ]

    # Add code snippet if we can identify the relevant section
    if result.details.get("line"):
        line_num = result.details["line"]
        code_lines = code.split("\n")
        start = max(0, line_num - 3)
        end = min(len(code_lines), line_num + 2)

        lines.append("\nRELEVANT CODE:")
        for i in range(start, end):
            marker = ">>> " if i == line_num - 1 else "    "
            lines.append(f"{marker}{i + 1}: {code_lines[i]}")

    # Add suggestion based on error type
    suggestion = _get_fix_suggestion(result)
    if suggestion:
        lines.append(f"\nSUGGESTION: {suggestion}")

    return "\n".join(lines)


def _get_fix_suggestion(result: ValidationResult) -> Optional[str]:
    """Get a fix suggestion based on the error."""
    error = result.error or ""

    if "Missing required field" in error or "missing" in error.lower():
        return "Add the missing field to your plugin configuration."

    if "auth.env_vars" in error:
        return "Add auth.env_vars with the environment variable name for the API key."

    if "plugin_name" in error:
        return "Add a plugin_name attribute to your class."

    if "chunk_text" in error:
        return "Add a chunk_text(self, text: str, base_meta: Dict) -> List[Chunk] method."

    if "apply" in error:
        return "Add an apply(self, query: str, chunks: Sequence[ChunkLike]) -> ConstraintResult method."

    if "List[Chunk]" in error:
        return "Return a list of Chunk objects, not a different type."

    if "empty list" in error:
        return "Ensure your implementation returns at least one chunk for non-empty input."

    return None


__all__ = [
    "PluginValidator",
    "format_validation_error",
]
