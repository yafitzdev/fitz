from __future__ import annotations

import typer
import yaml
from pathlib import Path
import importlib.metadata

from fitz_rag.config.loader import load_config
from fitz_rag.config.schema import RAGConfig
from fitz_rag.pipeline.engine import create_pipeline_from_yaml
from fitz_rag.vector_db.qdrant_client import create_qdrant_client

from fitz_rag.exceptions.base import FitzRAGError
from fitz_rag.exceptions.retriever import VectorSearchError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import CLI


app = typer.Typer(help="🔥 Fitz-RAG — Retrieval-Augmented Generation Toolkit")
logger = get_logger(__name__)


# ---------------------------------------------------------
# Version
# ---------------------------------------------------------
@app.command()
def version():
    """Show current Fitz-RAG version."""
    try:
        v = importlib.metadata.version("fitz_rag")
    except importlib.metadata.PackageNotFoundError:
        v = "0.1.0"
    typer.echo(f"fitz-rag version {v}")


# ---------------------------------------------------------
# Config: show & path
# ---------------------------------------------------------
@app.command("config-show")
def config_show():
    """Print the merged configuration (default + user)."""
    try:
        cfg = load_config()
        typer.echo(yaml.dump(cfg, sort_keys=False))
    except Exception as e:
        logger.error(f"{CLI} Failed to load configuration: {e}")
        raise typer.Exit(code=1)


@app.command("config-path")
def config_path():
    """Show the resolved config path."""
    # load_config(return_path=True) removed → we do explicit logic
    env = Path(typer.getenv("FITZ_RAG_CONFIG", "")) if typer.getenv("FITZ_RAG_CONFIG") else None

    if env and env.exists():
        typer.echo(str(env))
    else:
        # default.yaml inside package
        from importlib.resources import files
        default_path = files("fitz_rag.config").joinpath("default.yaml")
        typer.echo(str(default_path))


# ---------------------------------------------------------
# Collections (Qdrant)
# ---------------------------------------------------------
collections_app = typer.Typer(help="Manage Qdrant collections")
app.add_typer(collections_app, name="collections")


def _make_qdrant_client_from_config() -> object:
    """Internal helper: Build a Qdrant client using unified config."""
    raw = load_config()
    cfg = RAGConfig.from_dict(raw)

    return create_qdrant_client(
        host=cfg.retriever.qdrant_host,
        port=cfg.retriever.qdrant_port,
        https=False,
        api_key=None,
    )


@collections_app.command("list")
def collections_list():
    """List Qdrant collections."""
    try:
        client = _make_qdrant_client_from_config()
        cols = client.get_collections().collections
    except Exception as e:
        logger.error(f"{CLI} Qdrant connection failed: {e}")
        raise typer.Exit(code=1)

    if not cols:
        typer.echo("No collections found.")
        return

    for c in cols:
        typer.echo(f"- {c.name}")


@collections_app.command("drop")
def collections_drop(name: str):
    """Drop a Qdrant collection."""
    try:
        client = _make_qdrant_client_from_config()
    except Exception as e:
        logger.error(f"{CLI} Qdrant connection failed: {e}")
        raise typer.Exit(code=1)

    if typer.confirm(f"Delete collection '{name}'?"):
        try:
            client.delete_collection(name)
            typer.echo(f"Deleted collection: {name}")
        except Exception as e:
            logger.error(f"{CLI} Failed deleting collection: {e}")
            raise typer.Exit(code=1)


# ---------------------------------------------------------
# Query
# ---------------------------------------------------------
@app.command()
def query(text: str, config: str | None = typer.Option(None, help="Path to config YAML")):
    """
    Run full RAG pipeline:
      - load config
      - build pipeline
      - retrieve + generate
    """
    try:
        pipeline = create_pipeline_from_yaml(config)
        answer = pipeline.run(text)
    except FitzRAGError as e:
        logger.error(f"{CLI} Pipeline error: {e}")
        raise typer.Exit(code=1)
    except VectorSearchError as e:
        logger.error(f"{CLI} Qdrant error: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"{CLI} Unexpected error: {e}")
        raise typer.Exit(code=1)

    typer.echo("\n=== ANSWER ===")
    typer.echo(answer.answer)

    typer.echo("\n=== SOURCES ===")
    for src in answer.sources:
        typer.echo(f"- {src.source_id} | metadata={src.metadata}")


if __name__ == "__main__":
    app()
