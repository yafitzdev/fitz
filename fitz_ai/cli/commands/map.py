# fitz_ai/cli/commands/map.py
"""
Knowledge map visualization command.

Usage:
    fitz map                           # Generate and open knowledge map
    fitz map -o custom.html            # Custom output path
    fitz map --no-open                 # Don't open in browser
    fitz map --rebuild                 # Ignore cache, fetch all embeddings
    fitz map --similarity-threshold 0.7  # Lower threshold = more edges
"""

from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import ui
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Main Command
# =============================================================================


def command(
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output HTML file path. Default: .fitz/knowledge_map.html",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection to visualize. Uses config default if not specified.",
    ),
    no_open: bool = typer.Option(
        False,
        "--no-open",
        help="Don't open the HTML file in browser after generation.",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        help="Rebuild cache from scratch (ignore existing cached embeddings).",
    ),
    similarity_threshold: float = typer.Option(
        0.8,
        "--similarity-threshold",
        "-t",
        help="Minimum similarity for edges (0.0-1.0). Lower = more edges.",
    ),
    no_similarity_edges: bool = typer.Option(
        False,
        "--no-similarity-edges",
        help="Don't show similarity edges between chunks (faster for large datasets).",
    ),
) -> None:
    """
    Generate an interactive visualization of your knowledge base.

    Shows clusters of related content and gaps in coverage.
    Outputs a self-contained HTML file that can be opened in any browser.

    Examples:
        fitz map                    # Default visualization
        fitz map -o my_map.html     # Custom output file
        fitz map --rebuild          # Force rebuild cache
    """
    # =========================================================================
    # Check dependencies
    # =========================================================================

    try:
        import umap  # noqa: F401
    except ImportError:
        ui.error("umap-learn is required for knowledge map.")
        ui.info("Install with: pip install fitz-ai[map]")
        raise typer.Exit(1)

    try:
        from sklearn.cluster import KMeans  # noqa: F401
    except ImportError:
        ui.error("scikit-learn is required for knowledge map.")
        ui.info("Install with: pip install fitz-ai[map]")
        raise typer.Exit(1)

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Map", "Visualize your knowledge base")

    # =========================================================================
    # Load config via CLIContext
    # =========================================================================

    ctx = CLIContext.load_or_none()
    if ctx is None:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)

    embedding_id = ctx.embedding_id

    # =========================================================================
    # Determine collection
    # =========================================================================

    if collection is None:
        collection = ctx.retrieval_collection

        # Check if it exists
        collections = ctx.get_collections()
        if not collections:
            ui.error("No collections found. Run 'fitz ingest' first.")
            raise typer.Exit(1)

        if collection not in collections:
            if len(collections) == 1:
                collection = collections[0]
            else:
                collection = ui.prompt_numbered_choice(
                    "Select collection",
                    collections,
                    collections[0],
                )

    ui.info(f"Collection: {collection}")

    # =========================================================================
    # Determine output path
    # =========================================================================

    if output is None:
        output = FitzPaths.knowledge_map_html()

    # =========================================================================
    # Load or create state
    # =========================================================================

    from fitz_ai.map.state import KnowledgeMapStateManager

    state_manager = KnowledgeMapStateManager()

    if rebuild:
        ui.info("Rebuilding cache from scratch...")
        state = state_manager._create_new_state(collection, embedding_id)
        state_manager._state = state
        state_manager._dirty = True
    else:
        state = state_manager.load(collection, embedding_id)

    # =========================================================================
    # Fetch embeddings
    # =========================================================================

    ui.info("Fetching embeddings from vector DB...")

    from fitz_ai.map.embeddings import fetch_all_chunk_ids, fetch_chunk_embeddings

    vdb = ctx.get_vector_db_client()

    # Check if vector DB supports scroll_with_vectors
    if not hasattr(vdb, "scroll_with_vectors"):
        ui.error(
            f"Vector DB {type(vdb).__name__} does not support scroll_with_vectors. "
            "Knowledge map currently only supports FAISS backend."
        )
        raise typer.Exit(1)

    # Get all current chunk IDs
    current_chunk_ids = fetch_all_chunk_ids(vdb, collection)

    if not current_chunk_ids:
        ui.error(f"No chunks found in collection '{collection}'. Run 'fitz ingest' first.")
        raise typer.Exit(1)

    # Remove stale chunks from cache
    removed = state_manager.remove_stale_chunks(current_chunk_ids)
    if removed > 0:
        ui.info(f"Removed {removed} stale chunks from cache")

    # Determine which chunks need fetching
    chunks_to_fetch = state_manager.get_chunks_needing_fetch(current_chunk_ids)

    if chunks_to_fetch:
        ui.info(f"Fetching {len(chunks_to_fetch)} new embeddings...")
        new_embeddings = fetch_chunk_embeddings(vdb, collection, chunks_to_fetch)
        state_manager.add_chunks(new_embeddings)
        ui.success(f"Cached {len(new_embeddings)} new embeddings")
    else:
        ui.info("All embeddings already cached")

    # Get all chunks from state
    all_chunks = list(state.chunks.values())
    ui.info(f"Total chunks: {len(all_chunks)}")

    # =========================================================================
    # Run UMAP projection
    # =========================================================================

    ui.info("Running UMAP projection...")

    from fitz_ai.map.embeddings import embeddings_to_matrix
    from fitz_ai.map.projection import (
        assign_coordinates,
        compute_document_centroids,
        run_umap_projection,
    )

    matrix, chunk_ids = embeddings_to_matrix(all_chunks)
    coordinates = run_umap_projection(matrix)
    all_chunks = assign_coordinates(all_chunks, coordinates, chunk_ids)

    # Compute document centroids
    documents = compute_document_centroids(state.documents, all_chunks)

    # =========================================================================
    # Run clustering
    # =========================================================================

    ui.info("Detecting clusters...")

    from fitz_ai.map.clustering import (
        assign_cluster_labels,
        detect_clusters,
        extract_cluster_keywords,
    )

    cluster_labels, cluster_info = detect_clusters(coordinates, chunk_ids)
    all_chunks = assign_cluster_labels(all_chunks, cluster_labels, chunk_ids)

    # Extract keywords for each cluster
    for ci in cluster_info:
        ci.keywords = extract_cluster_keywords(all_chunks, ci.cluster_id, top_k=5)
        if ci.keywords:
            ci.label = ", ".join(ci.keywords[:3]).title()

    ui.info(f"Found {len(cluster_info)} clusters")

    # =========================================================================
    # Detect gaps
    # =========================================================================

    ui.info("Detecting gaps...")

    from fitz_ai.map.gaps import compute_coverage_score, detect_gaps, mark_gap_chunks

    gap_info, gap_chunk_ids = detect_gaps(all_chunks, cluster_info, coordinates, chunk_ids)
    all_chunks = mark_gap_chunks(all_chunks, gap_chunk_ids)

    coverage_score = compute_coverage_score(len(all_chunks), len(gap_chunk_ids))

    if gap_info:
        ui.warning(f"Found {len(gap_info)} gap regions with {len(gap_chunk_ids)} affected chunks")
    else:
        ui.success("No significant gaps detected")

    # =========================================================================
    # Compute stats
    # =========================================================================

    from fitz_ai.map.models import MapStats

    stats = MapStats(
        total_chunks=len(all_chunks),
        total_documents=len(documents),
        num_clusters=len(cluster_info),
        num_gaps=len(gap_info),
        avg_cluster_size=len(all_chunks) / max(1, len(cluster_info)),
        coverage_score=coverage_score,
    )

    # Update state with results
    state.clusters = cluster_info
    state.gaps = gap_info
    state.stats = stats

    # =========================================================================
    # Generate HTML
    # =========================================================================

    ui.info("Generating HTML visualization...")

    from fitz_ai.map.html_generator import generate_html

    generate_html(
        chunks=all_chunks,
        documents=list(documents.values()),
        clusters=cluster_info,
        gaps=gap_info,
        stats=stats,
        output_path=output,
        include_similarity_edges=not no_similarity_edges,
        similarity_threshold=similarity_threshold,
    )

    # =========================================================================
    # Save state
    # =========================================================================

    state_manager.save()

    # =========================================================================
    # Done
    # =========================================================================

    ui.success(f"Knowledge map generated: {output}")

    if not no_open:
        ui.info("Opening in browser...")
        webbrowser.open(f"file://{output.absolute()}")


__all__ = ["command"]
