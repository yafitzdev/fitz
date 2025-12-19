# fitz/cli/errors.py
# -*- coding: utf-8 -*-
"""
Global error handler for Fitz CLI.

Catches common exceptions and displays helpful, actionable error messages
instead of raw tracebacks.
"""

from __future__ import annotations

import functools
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import typer

# Rich for pretty output (optional)
try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console = Console(stderr=True)
except ImportError:
    RICH_AVAILABLE = False
    console = None


# =============================================================================
# Error Patterns and Fixes
# =============================================================================


@dataclass
class ErrorFix:
    """A suggested fix for an error."""

    title: str
    description: str
    commands: list[str] | None = None


@dataclass
class ErrorPattern:
    """Pattern matching for known errors."""

    pattern: str  # Regex pattern to match error message
    title: str
    description: str
    fixes: list[ErrorFix]
    category: str = "general"


# Known error patterns with fixes
ERROR_PATTERNS: list[ErrorPattern] = [
    # =========================================================================
    # Connection Errors
    # =========================================================================
    ErrorPattern(
        pattern=r"(Connection refused|Cannot connect|Failed to connect).*(:6333|qdrant)",
        title="Qdrant Connection Failed",
        description="Cannot connect to Qdrant vector database.",
        category="connection",
        fixes=[
            ErrorFix(
                "Start Qdrant with Docker",
                "Run Qdrant locally using Docker:",
                ["docker run -p 6333:6333 qdrant/qdrant"],
            ),
            ErrorFix(
                "Check Qdrant host",
                "If Qdrant is running elsewhere, set the environment variable:",
                [
                    "$env:QDRANT_HOST = 'your-host-ip'  # PowerShell",
                    "export QDRANT_HOST=your-host-ip    # Bash",
                ],
            ),
            ErrorFix(
                "Use local FAISS instead",
                "Run 'fitz init' and select FAISS as vector database",
                ["fitz init"],
            ),
        ],
    ),
    ErrorPattern(
        pattern=r"(Connection refused|Cannot connect).*(:11434|ollama)",
        title="Ollama Connection Failed",
        description="Cannot connect to Ollama local LLM server.",
        category="connection",
        fixes=[
            ErrorFix(
                "Install and start Ollama",
                "Download from https://ollama.com and run:",
                ["ollama serve"],
            ),
            ErrorFix(
                "Pull a model",
                "After starting Ollama, pull a model:",
                ["ollama pull llama3.2", "ollama pull nomic-embed-text"],
            ),
            ErrorFix(
                "Use cloud API instead",
                "Set an API key and run 'fitz init':",
                ["$env:COHERE_API_KEY = 'your-key'", "fitz init"],
            ),
        ],
    ),
    # =========================================================================
    # API Key Errors
    # =========================================================================
    ErrorPattern(
        pattern=r"(COHERE_API_KEY|Cohere).*(not set|missing|invalid)",
        title="Cohere API Key Missing",
        description="The COHERE_API_KEY environment variable is not set.",
        category="credentials",
        fixes=[
            ErrorFix(
                "Set the API key",
                "Get your key from https://dashboard.cohere.com/api-keys",
                [
                    "$env:COHERE_API_KEY = 'your-key'  # PowerShell",
                    "export COHERE_API_KEY=your-key    # Bash",
                ],
            ),
            ErrorFix(
                "Use local LLM instead",
                "Install Ollama for offline usage:",
                [
                    "# Visit https://ollama.com to install",
                    "ollama pull llama3.2",
                    "fitz init  # Select 'ollama' as provider",
                ],
            ),
        ],
    ),
    ErrorPattern(
        pattern=r"(OPENAI_API_KEY|OpenAI).*(not set|missing|invalid)",
        title="OpenAI API Key Missing",
        description="The OPENAI_API_KEY environment variable is not set.",
        category="credentials",
        fixes=[
            ErrorFix(
                "Set the API key",
                "Get your key from https://platform.openai.com/api-keys",
                [
                    "$env:OPENAI_API_KEY = 'sk-...'  # PowerShell",
                    "export OPENAI_API_KEY=sk-...    # Bash",
                ],
            ),
        ],
    ),
    ErrorPattern(
        pattern=r"(ANTHROPIC_API_KEY|Anthropic).*(not set|missing|invalid)",
        title="Anthropic API Key Missing",
        description="The ANTHROPIC_API_KEY environment variable is not set.",
        category="credentials",
        fixes=[
            ErrorFix(
                "Set the API key",
                "Get your key from https://console.anthropic.com/",
                [
                    "$env:ANTHROPIC_API_KEY = 'sk-ant-...'  # PowerShell",
                    "export ANTHROPIC_API_KEY=sk-ant-...    # Bash",
                ],
            ),
        ],
    ),
    # =========================================================================
    # Collection Errors
    # =========================================================================
    ErrorPattern(
        pattern=r"Collection '?([^']+)'? (not found|doesn't exist|does not exist)",
        title="Collection Not Found",
        description="The specified vector database collection does not exist.",
        category="data",
        fixes=[
            ErrorFix(
                "Ingest documents first",
                "Create the collection by ingesting documents:",
                ["fitz-ingest run ./your_docs --collection your_collection"],
            ),
            ErrorFix(
                "Check collection name",
                "List available collections:",
                ["fitz-ingest stats --collection default"],
            ),
        ],
    ),
    ErrorPattern(
        pattern=r"(No (documents|chunks)|empty).*(collection|result)",
        title="No Documents Found",
        description="The collection exists but contains no documents.",
        category="data",
        fixes=[
            ErrorFix(
                "Ingest documents",
                "Add documents to your collection:",
                ["fitz-ingest run ./your_docs --collection your_collection"],
            ),
            ErrorFix(
                "Check ingestion logs",
                "Verify documents were ingested successfully:",
                ["fitz-ingest validate ./your_docs"],
            ),
        ],
    ),
    # =========================================================================
    # Model Errors
    # =========================================================================
    ErrorPattern(
        pattern=r"model '?([^']+)'? (was removed|deprecated|not found|does not exist)",
        title="Model Not Available",
        description="The specified model is no longer available or doesn't exist.",
        category="model",
        fixes=[
            ErrorFix(
                "Update your config",
                "Edit your config file and update the model name:",
                [
                    "# For Cohere, try: command-r-08-2024",
                    "# For OpenAI, try: gpt-4o-mini",
                    "fitz init  # Or re-run setup wizard",
                ],
            ),
        ],
    ),
    ErrorPattern(
        pattern=r"(rate limit|too many requests|429)",
        title="API Rate Limit Exceeded",
        description="You've exceeded the API rate limit.",
        category="api",
        fixes=[
            ErrorFix("Wait and retry", "Wait a few minutes before retrying.", []),
            ErrorFix(
                "Use local LLM",
                "Switch to Ollama for unlimited local usage:",
                ["ollama pull llama3.2", "fitz init"],
            ),
        ],
    ),
    # =========================================================================
    # Config Errors
    # =========================================================================
    ErrorPattern(
        pattern=r"(Config|Configuration).*(not found|missing|invalid)",
        title="Configuration Error",
        description="The configuration file is missing or invalid.",
        category="config",
        fixes=[
            ErrorFix("Run setup wizard", "Create a new configuration:", ["fitz init"]),
            ErrorFix(
                "Check config syntax", "View your current config:", ["fitz-pipeline config show"]
            ),
        ],
    ),
    # =========================================================================
    # Vector/Embedding Errors
    # =========================================================================
    ErrorPattern(
        pattern=r"(dimension|size) mismatch|expected (\d+).*(got|received) (\d+)",
        title="Vector Dimension Mismatch",
        description="The embedding dimensions don't match the collection configuration.",
        category="data",
        fixes=[
            ErrorFix(
                "Recreate collection",
                "Delete and recreate the collection with new embeddings:",
                [
                    "# Delete collection first (via Qdrant dashboard or API)",
                    "fitz-ingest run ./your_docs --collection new_collection",
                ],
            ),
            ErrorFix(
                "Use matching embedding model",
                "Ensure you're using the same embedding model for ingest and query.",
                [],
            ),
        ],
    ),
    ErrorPattern(
        pattern=r"vector name.*(error|mismatch|not found)",
        title="Vector Name Configuration Error",
        description="The collection uses named vectors but the code expects unnamed (or vice versa).",
        category="config",
        fixes=[
            ErrorFix(
                "Recreate collection",
                "Delete and recreate with simple vector configuration:",
                ["fitz-ingest run ./docs --collection new_collection"],
            ),
        ],
    ),
    # =========================================================================
    # File Errors
    # =========================================================================
    ErrorPattern(
        pattern=r"(File|Path|Directory).*(not found|does not exist|No such file)",
        title="File Not Found",
        description="The specified file or directory does not exist.",
        category="file",
        fixes=[
            ErrorFix(
                "Check the path",
                "Verify the file/directory exists:",
                ["ls ./your_path  # or 'dir' on Windows"],
            ),
        ],
    ),
    ErrorPattern(
        pattern=r"(Permission denied|Access denied)",
        title="Permission Denied",
        description="You don't have permission to access this file or directory.",
        category="file",
        fixes=[
            ErrorFix(
                "Check permissions",
                "Run with appropriate permissions or change file ownership.",
                [],
            ),
        ],
    ),
]


# =============================================================================
# Error Display Functions
# =============================================================================


def format_error_message(
    title: str,
    description: str,
    fixes: list[ErrorFix],
    original_error: str | None = None,
    show_traceback: bool = False,
) -> str:
    """Format a user-friendly error message."""
    lines = []
    lines.append(f"\n❌ {title}\n")
    lines.append(f"{description}\n")

    if fixes:
        lines.append("\nHow to fix:\n")
        for i, fix in enumerate(fixes, 1):
            lines.append(f"{i}. {fix.title}")
            lines.append(f"   {fix.description}")
            if fix.commands:
                for cmd in fix.commands:
                    lines.append(f"   $ {cmd}")
            lines.append("")

    if original_error and (show_traceback or os.getenv("FITZ_DEBUG")):
        lines.append(f"\nOriginal error: {original_error}")

    return "\n".join(lines)


def display_error(
    title: str,
    description: str,
    fixes: list[ErrorFix],
    original_error: str | None = None,
) -> None:
    """Display an error message to the user."""
    show_traceback = bool(os.getenv("FITZ_DEBUG"))

    if RICH_AVAILABLE and console:
        # Build rich panel content
        content_lines = [description, ""]

        if fixes:
            content_lines.append("[bold]How to fix:[/bold]")
            content_lines.append("")
            for i, fix in enumerate(fixes, 1):
                content_lines.append(f"[bold]{i}. {fix.title}[/bold]")
                content_lines.append(f"   {fix.description}")
                if fix.commands:
                    for cmd in fix.commands:
                        content_lines.append(f"   [dim]$[/dim] [cyan]{cmd}[/cyan]")

        content = "\n".join(content_lines)

        console.print(
            Panel(
                content,
                title=f"❌ {title}",
                border_style="red",
                expand=False,
            )
        )

        if original_error and show_traceback:
            console.print(f"\n[dim]Original error: {original_error}[/dim]")
    else:
        # Plain text fallback
        print(format_error_message(title, description, fixes, original_error, show_traceback))


# =============================================================================
# Error Matching
# =============================================================================


def match_error(error_message: str) -> ErrorPattern | None:
    """Match an error message against known patterns."""
    for pattern in ERROR_PATTERNS:
        if re.search(pattern.pattern, error_message, re.IGNORECASE):
            return pattern
    return None


def handle_import_error(exc: ImportError | ModuleNotFoundError) -> None:
    """Handle import errors with helpful messages showing the missing package."""
    # Extract the module name from the exception
    module_name = getattr(exc, "name", None) or str(exc)

    # Try to extract from the message if name attr is not set
    if not module_name or module_name == str(exc):
        match = re.search(r"No module named ['\"]?([^'\"]+)['\"]?", str(exc))
        if match:
            module_name = match.group(1)

    # Map common module names to pip package names
    pip_name_map = {
        "yaml": "pyyaml",
        "cv2": "opencv-python",
        "PIL": "pillow",
        "sklearn": "scikit-learn",
    }

    # Get the root module name (e.g., 'fitz.foo.bar' -> 'fitz')
    root_module = module_name.split(".")[0] if module_name else "unknown"

    # NEW: Check if this is an internal fitz module error
    if root_module == "fitz":
        # This is an internal import error, not a missing package
        # Show the actual traceback for debugging
        tb = traceback.format_exc()
        display_error(
            title=f"Internal Import Error: {module_name}",
            description=f"Failed to import internal module '{module_name}'. This is likely a code issue, not a missing package.",
            fixes=[
                ErrorFix(
                    "Check for circular imports",
                    "Run with debug mode to see the full traceback:",
                    [
                        "$env:FITZ_DEBUG = '1'  # PowerShell",
                        "export FITZ_DEBUG=1    # Bash",
                        "fitz doctor",
                    ],
                ),
                ErrorFix(
                    "Reinstall fitz in development mode",
                    "Make sure fitz is properly installed:",
                    ["pip install -e ."],
                ),
                ErrorFix(
                    "Test the import directly",
                    "Debug the specific import:",
                    [f'python -c "from {module_name} import *"'],
                ),
            ],
            original_error=tb,  # Always show traceback for internal errors
        )
        return

    pip_name = pip_name_map.get(root_module, root_module)

    display_error(
        title=f"Missing Dependency: {module_name}",
        description=f"The Python package '{module_name}' is not installed.",
        fixes=[
            ErrorFix(
                "Install the package",
                "Install the missing dependency:",
                [f"pip install {pip_name}"],
            ),
            ErrorFix(
                "Install all fitz dependencies",
                "Reinstall fitz with all dependencies:",
                ["pip install -e .", "# or: pip install -e .[all]"],
            ),
        ],
        original_error=str(exc),
    )


def handle_exception(exc: Exception) -> None:
    """Handle an exception with a user-friendly message."""
    # Special handling for import errors
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        handle_import_error(exc)
        return

    error_str = str(exc)
    error_type = type(exc).__name__

    # Get full traceback for debugging
    tb = traceback.format_exc()

    # Try to match a known error pattern
    pattern = match_error(error_str) or match_error(tb)

    if pattern:
        display_error(
            title=pattern.title,
            description=pattern.description,
            fixes=pattern.fixes,
            original_error=f"{error_type}: {error_str}",
        )
    else:
        # Unknown error - show generic message with the error
        display_error(
            title=f"Unexpected Error: {error_type}",
            description=error_str,
            fixes=[
                ErrorFix(
                    "Check the logs",
                    "Run with FITZ_DEBUG=1 for more details:",
                    ["$env:FITZ_DEBUG = '1'  # PowerShell", "export FITZ_DEBUG=1    # Bash"],
                ),
                ErrorFix("Run diagnostics", "Check your setup:", ["fitz doctor"]),
                ErrorFix(
                    "Report a bug",
                    "If this seems like a bug, please report it at:",
                    ["https://github.com/yafitzdev/fitz/issues"],
                ),
            ],
            original_error=tb if os.getenv("FITZ_DEBUG") else f"{error_type}: {error_str}",
        )


# =============================================================================
# Decorator for CLI Commands
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def friendly_errors(func: F) -> F:
    """
    Decorator that wraps CLI commands with user-friendly error handling.

    Usage:
        @friendly_errors
        def my_command():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except typer.Exit:
            # Let typer exits pass through
            raise
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            raise typer.Exit(code=130)
        except (ImportError, ModuleNotFoundError) as exc:
            handle_import_error(exc)
            raise typer.Exit(code=1)
        except Exception as exc:
            handle_exception(exc)
            raise typer.Exit(code=1)

    return wrapper  # type: ignore


def install_global_handler() -> None:
    """
    Install a global exception handler for unhandled exceptions.

    Call this at CLI startup to catch any exceptions that escape.
    """
    original_excepthook = sys.excepthook

    def custom_excepthook(exc_type: type, exc_value: BaseException, exc_tb: Any) -> None:
        if exc_type is KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            sys.exit(130)
        elif issubclass(exc_type, (ImportError, ModuleNotFoundError)):
            handle_import_error(exc_value)  # type: ignore
            sys.exit(1)
        elif issubclass(exc_type, Exception):
            handle_exception(exc_value)  # type: ignore
            sys.exit(1)
        else:
            # For non-Exception types (like SystemExit), use original
            original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = custom_excepthook


# =============================================================================
# Convenience function for testing
# =============================================================================


def test_error_matching():
    """Test error pattern matching with sample errors."""
    test_errors = [
        "Connection refused: localhost:6333",
        "COHERE_API_KEY is not set",
        "Collection 'default' not found",
        "model 'command-r-plus' was removed",
        "rate limit exceeded",
        "dimension mismatch: expected 1024, got 768",
        "No module named 'qdrant_client'",
    ]

    print("Testing error pattern matching:\n")
    for error in test_errors:
        pattern = match_error(error)
        if pattern:
            print(f"✓ '{error[:40]}...'")
            print(f"  → {pattern.title}")
        else:
            print(f"✗ '{error[:40]}...'")
            print("  → No match")
        print()


if __name__ == "__main__":
    test_error_matching()
