# fitz_sage/cli/commands/serve.py
"""
API server command.

Usage:
    fitz serve              # Start on default port 8000
    fitz serve --port 3000  # Custom port
    fitz serve --host 0.0.0.0  # Listen on all interfaces
"""

from __future__ import annotations

import typer

from fitz_sage.cli.ui import ui
from fitz_sage.logging.logger import get_logger

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
        ui.error("uvicorn not installed. Install with: pip install fitz-sage[api]")
        raise typer.Exit(1)

    try:
        from fitz_sage.api import create_app
    except ImportError as e:
        ui.error(f"Failed to import API module: {e}")
        ui.info("Install with: pip install fitz-sage[api]")
        raise typer.Exit(1)

    # Require config before starting server (no interactive prompt in server mode)
    from fitz_sage.config.firstrun import needs_firstrun
    from fitz_sage.core.paths import FitzPaths

    if needs_firstrun():
        config_path = FitzPaths.config()
        ui.error("No configuration found.")
        ui.info("Run a query first to auto-configure:")
        ui.info('  fitz query "test" --source ./docs')
        ui.info("Or create the config manually:")
        ui.info(f"  {config_path}")
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
