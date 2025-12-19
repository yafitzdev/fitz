"""
Validate command: Check documents before full ingestion (dry-run).

Usage:
    fitz-ingest validate ./documents
    fitz-ingest validate ./documents --ingest-plugin local
    fitz-ingest validate ./documents --show-errors
"""

from pathlib import Path

import typer

from fitz_ai.ingest.ingestion.engine import IngestionEngine
from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
from fitz_ai.ingest.validation.documents import ValidationConfig, validate
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import CLI, INGEST

logger = get_logger(__name__)


def command(
    source: Path = typer.Argument(
        ...,
        help="Source to validate (file or directory).",
    ),
    ingest_plugin: str = typer.Option(
        "local",
        "--ingest-plugin",
        "-i",
        help="Ingestion plugin to use for reading documents.",
    ),
    show_errors: bool = typer.Option(
        False,
        "--show-errors",
        help="Show detailed error messages for invalid documents.",
    ),
    min_chars: int = typer.Option(
        10,
        "--min-chars",
        help="Minimum character count for valid documents.",
    ),
) -> None:
    """
    Validate documents before full ingestion (dry-run).

    This command checks your documents without actually ingesting them:
    - Counts total documents found
    - Identifies empty or invalid documents
    - Shows which files would be filtered out
    - Estimates chunk count

    Useful for:
    - Testing your document source before full ingestion
    - Identifying problematic files
    - Estimating ingestion size/cost

    Examples:
        # Basic validation
        fitz-ingest validate ./docs

        # Show detailed errors
        fitz-ingest validate ./docs --show-errors

        # Custom minimum character count
        fitz-ingest validate ./docs --min-chars 50
    """
    # Validate source path
    if not source.exists():
        typer.echo(f"ERROR: source does not exist: {source}")
        raise typer.Exit(code=1)

    if not (source.is_file() or source.is_dir()):
        typer.echo(f"ERROR: source is not file or directory: {source}")
        raise typer.Exit(code=1)

    logger.info(f"{CLI}{INGEST} Validating source: {source}")

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("DOCUMENT VALIDATION (DRY-RUN)")
    typer.echo("=" * 60)
    typer.echo()

    # 1) Ingest documents
    typer.echo(f"[1/2] Reading documents from {source}...")
    IngestPluginCls = get_ingest_plugin(ingest_plugin)
    ingest_plugin_obj = IngestPluginCls()
    ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})

    raw_docs = list(ingest_engine.run(str(source)))
    typer.echo(f"  ✓ Found {len(raw_docs)} documents")

    # 2) Validate documents
    typer.echo()
    typer.echo(f"[2/2] Validating documents (min_chars={min_chars})...")

    validation_config = ValidationConfig(
        min_chars=min_chars,
        strip_whitespace=True,
    )

    valid_docs = validate(raw_docs, validation_config)
    invalid_count = len(raw_docs) - len(valid_docs)

    typer.echo(f"  ✓ Valid documents: {len(valid_docs)}")
    if invalid_count > 0:
        typer.echo(f"  ✗ Invalid documents: {invalid_count}")

    # Show invalid documents if requested
    if show_errors and invalid_count > 0:
        typer.echo()
        typer.echo("Invalid documents:")
        typer.echo("-" * 40)
        invalid_docs = set(d.path for d in raw_docs) - set(d.path for d in valid_docs)
        for path in sorted(invalid_docs):
            doc = next(d for d in raw_docs if d.path == path)
            content_len = len(getattr(doc, "content", "") or getattr(doc, "text", ""))
            typer.echo(f"  • {path}")
            typer.echo(f"    Reason: Content length ({content_len}) < minimum ({min_chars})")

    # Show summary
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("VALIDATION SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"Total documents:   {len(raw_docs)}")
    typer.echo(f"Valid documents:   {len(valid_docs)}")
    typer.echo(f"Invalid documents: {invalid_count}")

    if len(valid_docs) > 0:
        # Estimate chunks (1:1 for now, could be more sophisticated)
        typer.echo(f"Estimated chunks:  ~{len(valid_docs)}")

        # Estimate content size
        total_chars = sum(
            len(getattr(d, "content", "") or getattr(d, "text", "")) for d in valid_docs
        )
        typer.echo(f"Total content:     ~{total_chars:,} characters")

    typer.echo()

    if invalid_count > 0:
        typer.echo("⚠️  Some documents were filtered out.")
        typer.echo("   Use --show-errors to see details.")
    else:
        typer.echo("✓ All documents passed validation!")

    typer.echo()
    typer.echo("💡 Next step: Run 'fitz-ingest run' to perform actual ingestion")
    typer.echo()
