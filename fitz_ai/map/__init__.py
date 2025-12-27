# fitz_ai/map/__init__.py
"""
Knowledge map visualization module.

This module provides tools for visualizing the knowledge base as an interactive
HTML graph, showing clusters of related content and gaps in coverage.

Usage:
    from fitz_ai.map import generate_knowledge_map

    # Generate visualization
    generate_knowledge_map(collection="default", output_path="map.html")
"""

from fitz_ai.map.models import (
    ChunkEmbedding,
    ClusterInfo,
    DocumentNode,
    GapInfo,
    KnowledgeMapState,
    MapStats,
)

__all__ = [
    "ChunkEmbedding",
    "DocumentNode",
    "ClusterInfo",
    "GapInfo",
    "MapStats",
    "KnowledgeMapState",
]
