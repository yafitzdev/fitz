"""
Test command: Verify pipeline setup and connections.

Usage:
    fitz-pipeline test
    fitz-pipeline test --config my_config.yaml
"""

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.engines.classic_rag.pipeline.pipeline.engine import create_pipeline_from_yaml
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
    skip_query: bool = typer.Option(
        False,
        "--skip-query",
        help="Skip running test query (only check connections)",
    ),
) -> None:
    """
    Test pipeline setup and verify all connections work.

    This command checks:
    1. Configuration loads successfully
    2. Pipeline can be built
    3. Components are properly configured
    4. (Optional) Run a simple test query

    Useful for:
    - Verifying setup after installation
    - Debugging connection issues
    - Pre-deployment validation

    Examples:
        # Full test with test query
        fitz-pipeline test

        # Quick connection check only
        fitz-pipeline test --skip-query

        # Test custom configuration
        fitz-pipeline test --config production.yaml
    """
    logger.info(
        f"{CLI}{PIPELINE} Testing pipeline with "
        f"config={config if config is not None else '<default>'}"
    )

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("PIPELINE TEST")
    typer.echo("=" * 60)
    typer.echo()

    # Test 1: Configuration loading
    typer.echo("[1/3] Testing configuration loading...")
    try:
        pipeline = create_pipeline_from_yaml(str(config) if config is not None else None)
        typer.echo("  ✓ Configuration loaded successfully")
    except Exception as e:
        typer.echo(f"  ✗ Configuration failed: {e}")
        raise typer.Exit(code=1)

    # Test 2: Pipeline components
    typer.echo()
    typer.echo("[2/3] Checking pipeline components...")
    try:
        # Check if pipeline has required components
        has_retriever = hasattr(pipeline, "_retriever") or hasattr(pipeline, "retriever")
        has_llm = hasattr(pipeline, "_llm") or hasattr(pipeline, "llm")
        has_rgs = hasattr(pipeline, "_rgs") or hasattr(pipeline, "rgs")

        if has_retriever:
            typer.echo("  ✓ Retriever configured")
        else:
            typer.echo("  ⚠ Retriever not found (may be optional)")

        if has_llm:
            typer.echo("  ✓ LLM configured")
        else:
            typer.echo("  ⚠ LLM not found (may be optional)")

        if has_rgs:
            typer.echo("  ✓ RGS (prompt builder) configured")
        else:
            typer.echo("  ⚠ RGS not found (may be optional)")

    except Exception as e:
        typer.echo(f"  ✗ Component check failed: {e}")
        raise typer.Exit(code=1)

    # Test 3: Optional test query
    if not skip_query:
        typer.echo()
        typer.echo("[3/3] Running test query...")
        try:
            test_question = "Hello, this is a test query."
            result = pipeline.run(test_question)

            if result and getattr(result, "answer", None):
                typer.echo("  ✓ Test query completed successfully")
                typer.echo(f"  Answer preview: {getattr(result, 'answer', '')[:100]}...")
            else:
                typer.echo("  ⚠ Test query completed but no answer generated")
        except Exception as e:
            typer.echo(f"  ✗ Test query failed: {e}")
            typer.echo()
            typer.echo("Possible issues:")
            typer.echo("  • LLM API key not set or invalid")
            typer.echo("  • Vector database not running")
            typer.echo("  • No documents in collection")
            raise typer.Exit(code=1)
    else:
        typer.echo()
        typer.echo("[3/3] Skipping test query (--skip-query)")

    # Summary
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("✓ PIPELINE TEST PASSED")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo("Your pipeline is configured correctly and ready to use!")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo('  • Run queries: fitz-pipeline query "Your question"')
    typer.echo("  • Check config: fitz-pipeline config show")
    typer.echo()
