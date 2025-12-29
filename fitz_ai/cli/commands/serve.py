# fitz_ai/cli/commands/serve.py
"""
API server command.

Usage:
    fitz serve              # Start on default port 8000
    fitz serve --port 3000  # Custom port
    fitz serve --host 0.0.0.0  # Listen on all interfaces
"""

from __future__ import annotations

import typer

from fitz_ai.cli.ui import ui
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


def command(
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host to bind to.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to listen on.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        "-r",
        help="Enable auto-reload for development.",
    ),
) -> None:
    """
    Start the Fitz API server.

    Runs a REST API server for programmatic access to Fitz.

    Examples:
        fitz serve                    # Start on localhost:8000
        fitz serve -p 3000            # Custom port
        fitz serve --host 0.0.0.0     # Listen on all interfaces
        fitz serve --reload           # Auto-reload on code changes

    API Documentation:
        Once running, visit http://localhost:8000/docs for interactive docs.
    """
    try:
        import uvicorn
    except ImportError:
        ui.error("uvicorn not installed. Install with: pip install fitz-ai[api]")
        raise typer.Exit(1)

    try:
        from fitz_ai.api import create_app
    except ImportError as e:
        ui.error(f"Failed to import API module: {e}")
        ui.info("Install with: pip install fitz-ai[api]")
        raise typer.Exit(1)

    ui.header("Fitz API Server", f"http://{host}:{port}")
    ui.info(f"API docs: http://{host}:{port}/docs")
    ui.info("Press Ctrl+C to stop")
    print()

    # Create app and run
    app = create_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
