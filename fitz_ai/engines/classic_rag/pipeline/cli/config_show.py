"""
Config show command: Display resolved pipeline configuration.

Usage:
    fitz-pipeline config show
    fitz-pipeline config show --config my_config.yaml
"""

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.engines.classic_rag.config.loader import load_config
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import CLI, PIPELINE

logger = get_logger(__name__)


def command(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to pipeline YAML config.",
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml, json",
    ),
) -> None:
    """
    Display the resolved pipeline configuration.

    Shows the complete configuration after:
    - Loading from YAML file (or using defaults)
    - Resolving environment variables
    - Applying defaults

    Useful for:
    - Debugging configuration issues
    - Verifying which plugins are being used
    - Checking credential resolution

    Examples:
        # Show default configuration
        fitz-pipeline config show

        # Show custom configuration
        fitz-pipeline config show --config my_config.yaml

        # Show as JSON (for scripting)
        fitz-pipeline config show --format json
    """
    logger.info(
        f"{CLI}{PIPELINE} Showing pipeline config from "
        f"{config if config is not None else '<default>'}"
    )

    # Load configuration
    raw_cfg = load_config(str(config) if config is not None else None)

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("PIPELINE CONFIGURATION")
    typer.echo("=" * 60)
    typer.echo()

    # Display based on format
    if format == "json":
        import json

        typer.echo(json.dumps(raw_cfg, indent=2))
    else:
        # Default to YAML-like display
        import yaml

        typer.echo(yaml.dump(raw_cfg, default_flow_style=False, sort_keys=False))

    typer.echo()
    typer.echo("=" * 60)
    typer.echo()
    typer.echo("💡 Tip: Use this to verify your plugin selections and credentials")
    typer.echo()
