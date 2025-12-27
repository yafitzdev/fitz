# fitz_ai/map/html_generator.py
"""
Generate interactive HTML visualization using vis.js.

Creates a self-contained HTML file with embedded CSS/JS for
visualizing the knowledge base as an interactive graph.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from fitz_ai.map.clustering import CLUSTER_COLORS
from fitz_ai.map.embeddings import compute_similarity_matrix, embeddings_to_matrix
from fitz_ai.map.models import (
    ChunkEmbedding,
    ClusterInfo,
    DocumentNode,
    GapInfo,
    MapStats,
)

logger = logging.getLogger(__name__)


def generate_html(
    chunks: List[ChunkEmbedding],
    documents: List[DocumentNode],
    clusters: List[ClusterInfo],
    gaps: List[GapInfo],
    stats: MapStats,
    output_path: Path,
    title: str = "Fitz Knowledge Map",
    include_similarity_edges: bool = True,
    similarity_threshold: float = 0.8,
    max_similarity_edges_per_node: int = 3,
) -> None:
    """
    Generate self-contained HTML file with vis.js visualization.

    Args:
        chunks: Chunks with coordinates and cluster assignments.
        documents: Document nodes with coordinates.
        clusters: Cluster information.
        gaps: Gap information.
        stats: Overall statistics.
        output_path: Where to write the HTML file.
        title: Page title.
        include_similarity_edges: Whether to show similarity edges.
        similarity_threshold: Minimum similarity for edges.
        max_similarity_edges_per_node: Maximum edges per node.
    """
    # Generate nodes and edges JSON
    nodes_json = generate_nodes_json(chunks, documents)

    edges_json = generate_edges_json(
        chunks,
        documents,
        include_similarity_edges=include_similarity_edges,
        similarity_threshold=similarity_threshold,
        max_edges_per_node=max_similarity_edges_per_node,
    )

    clusters_json = json.dumps([c.model_dump() for c in clusters])
    gaps_json = json.dumps([g.model_dump() for g in gaps])
    stats_json = json.dumps(stats.model_dump())

    # Generate HTML
    html = HTML_TEMPLATE.format(
        title=title,
        nodes_json=nodes_json,
        edges_json=edges_json,
        clusters_json=clusters_json,
        gaps_json=gaps_json,
        stats_json=stats_json,
    )

    # Write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    logger.info(f"Generated knowledge map at {output_path}")


def generate_nodes_json(
    chunks: List[ChunkEmbedding],
    documents: List[DocumentNode],
) -> str:
    """Generate vis.js nodes array as JSON."""
    nodes = []

    # Scale factor for positioning (vis.js works better with larger coordinates)
    scale = 400

    # Add document nodes
    for doc in documents:
        if doc.x is None or doc.y is None:
            continue

        nodes.append(
            {
                "id": f"doc:{doc.doc_id}",
                "label": doc.label[:30],
                "x": doc.x * scale,
                "y": doc.y * scale,
                "shape": "box",
                "color": {
                    "background": "#21262d",
                    "border": "#8b949e",
                    "highlight": {"background": "#30363d", "border": "#c9d1d9"},
                },
                "font": {"color": "#c9d1d9", "size": 12},
                "borderWidth": 2,
                "margin": 10,
                "nodeType": "document",
                "title": doc.doc_id,
            }
        )

    # Add chunk nodes
    for chunk in chunks:
        if chunk.x is None or chunk.y is None:
            continue

        color = _get_chunk_color(chunk)

        nodes.append(
            {
                "id": chunk.chunk_id,
                "label": chunk.label[:20],
                "x": chunk.x * scale,
                "y": chunk.y * scale,
                "shape": "dot",
                "size": 12 if chunk.is_gap else 10,
                "color": color,
                "font": {"color": "#8b949e", "size": 9},
                "nodeType": "chunk",
                "cluster_id": chunk.cluster_id,
                "is_gap": chunk.is_gap,
                "doc_id": chunk.doc_id,
                "content_preview": chunk.content_preview,
                "title": f"{chunk.label}\n\n{chunk.content_preview[:100]}...",
            }
        )

    return json.dumps(nodes)


def _get_chunk_color(chunk: ChunkEmbedding) -> dict:
    """Get color configuration for a chunk node."""
    if chunk.is_gap:
        return {
            "background": "#f85149",
            "border": "#f85149",
            "highlight": {"background": "#ff6b6b", "border": "#ffffff"},
        }

    if chunk.cluster_id is not None:
        base_color = CLUSTER_COLORS[chunk.cluster_id % len(CLUSTER_COLORS)]
    else:
        base_color = "#58a6ff"

    return {
        "background": base_color,
        "border": base_color,
        "highlight": {"background": base_color, "border": "#ffffff"},
    }


def generate_edges_json(
    chunks: List[ChunkEmbedding],
    documents: List[DocumentNode],
    include_similarity_edges: bool = True,
    similarity_threshold: float = 0.8,
    max_edges_per_node: int = 3,
) -> str:
    """
    Generate vis.js edges array as JSON.

    Edge types:
    - hierarchy: document -> chunk (solid line)
    - similarity: chunk <-> chunk (dashed line, if enabled)
    """
    edges = []

    # Hierarchy edges (document -> chunk)
    for doc in documents:
        for chunk_id in doc.chunk_ids:
            edges.append(
                {
                    "from": f"doc:{doc.doc_id}",
                    "to": chunk_id,
                    "color": {"color": "#30363d", "opacity": 0.5},
                    "width": 1,
                    "smooth": {"type": "continuous"},
                    "edgeType": "hierarchy",
                }
            )

    # Similarity edges
    if include_similarity_edges and chunks:
        similarity_edges = _compute_similarity_edges(
            chunks, similarity_threshold, max_edges_per_node
        )
        edges.extend(similarity_edges)

    return json.dumps(edges)


def _compute_similarity_edges(
    chunks: List[ChunkEmbedding],
    threshold: float,
    max_per_node: int,
) -> List[dict]:
    """Compute similarity edges between chunks."""
    edges = []

    # Convert embeddings to matrix
    matrix, chunk_ids = embeddings_to_matrix(chunks)

    if matrix.size == 0:
        return edges

    # Compute similarity matrix
    similarity = compute_similarity_matrix(matrix, threshold=threshold)

    # Track edges per node
    edge_count = {cid: 0 for cid in chunk_ids}

    # Find edges above threshold
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            sim = similarity[i, j]
            if sim >= threshold:
                # Check edge limits
                if edge_count[chunk_ids[i]] >= max_per_node:
                    continue
                if edge_count[chunk_ids[j]] >= max_per_node:
                    continue

                edges.append(
                    {
                        "from": chunk_ids[i],
                        "to": chunk_ids[j],
                        "color": {
                            "color": f"rgba(63, 185, 80, {float(sim) * 0.6})",
                            "highlight": "#3fb950",
                        },
                        "width": max(1, (float(sim) - threshold) * 5),
                        "smooth": {"type": "continuous"},
                        "edgeType": "similarity",
                        "similarity": float(sim),
                    }
                )

                edge_count[chunk_ids[i]] += 1
                edge_count[chunk_ids[j]] += 1

    return edges


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            height: 100vh;
            display: flex;
        }}
        #sidebar {{
            width: 320px;
            background: #161b22;
            border-right: 1px solid #30363d;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }}
        #graph {{ flex: 1; height: 100vh; }}
        h1 {{ font-size: 1.4rem; margin-bottom: 8px; color: #58a6ff; }}
        .subtitle {{ font-size: 0.85rem; color: #8b949e; margin-bottom: 20px; }}
        .section {{ margin-bottom: 24px; }}
        .section-title {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #8b949e;
            margin-bottom: 12px;
        }}
        .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
        .stat-box {{ background: #21262d; border-radius: 6px; padding: 12px; }}
        .stat-value {{ font-size: 1.5rem; font-weight: 600; color: #f0f6fc; }}
        .stat-label {{ font-size: 0.75rem; color: #8b949e; }}
        .cluster-item {{
            background: #21262d;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .cluster-item:hover {{ background: #30363d; }}
        .cluster-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .cluster-info {{ flex: 1; }}
        .cluster-name {{ font-weight: 500; margin-bottom: 2px; }}
        .cluster-count {{ font-size: 0.75rem; color: #8b949e; }}
        .gap-item {{
            background: #f8514926;
            border: 1px solid #f8514966;
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }}
        .gap-icon {{ color: #f85149; margin-right: 6px; }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
        .legend-line {{ width: 24px; height: 2px; }}
        #detail-panel {{
            background: #21262d;
            border-radius: 6px;
            padding: 12px;
            display: none;
        }}
        #detail-panel.active {{ display: block; }}
        .detail-title {{ font-weight: 600; margin-bottom: 8px; }}
        .detail-meta {{ font-size: 0.8rem; color: #8b949e; margin-bottom: 8px; }}
        .detail-content {{
            background: #0d1117;
            border-radius: 6px;
            padding: 10px;
            font-size: 0.85rem;
            line-height: 1.5;
            max-height: 150px;
            overflow-y: auto;
            border: 1px solid #30363d;
        }}
        .coverage-bar {{
            height: 8px;
            background: #21262d;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }}
        .coverage-fill {{
            height: 100%;
            background: linear-gradient(90deg, #f85149, #f0883e, #3fb950);
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>Knowledge Map</h1>
        <p class="subtitle">fitz-ai knowledge base visualization</p>

        <div class="section">
            <div class="section-title">Overview</div>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-value" id="stat-docs">0</div>
                    <div class="stat-label">Documents</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="stat-chunks">0</div>
                    <div class="stat-label">Chunks</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="stat-clusters">0</div>
                    <div class="stat-label">Clusters</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value" id="stat-gaps">0</div>
                    <div class="stat-label">Gaps</div>
                </div>
            </div>
            <div style="margin-top: 12px;">
                <div class="stat-label">Coverage Score</div>
                <div class="coverage-bar">
                    <div class="coverage-fill" id="coverage-bar" style="width: 100%"></div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Legend</div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #8b949e; border: 2px solid #8b949e;"></div>
                <span>Document</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #58a6ff;"></div>
                <span>Chunk</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: #f85149;"></div>
                <span>Gap (isolated)</span>
            </div>
            <div class="legend-item">
                <div class="legend-line" style="background: #30363d;"></div>
                <span>Hierarchy</span>
            </div>
            <div class="legend-item">
                <div class="legend-line" style="background: #3fb95066;"></div>
                <span>Similarity</span>
            </div>
        </div>

        <div class="section" id="clusters-section">
            <div class="section-title">Clusters</div>
            <div id="clusters-list"></div>
        </div>

        <div class="section" id="gaps-section">
            <div class="section-title">Detected Gaps</div>
            <div id="gaps-list"></div>
        </div>

        <div class="section">
            <div class="section-title">Selected Node</div>
            <div id="detail-panel">
                <div class="detail-title" id="detail-title"></div>
                <div class="detail-meta" id="detail-meta"></div>
                <div class="detail-content" id="detail-content"></div>
            </div>
            <div id="detail-placeholder" style="color: #484f58; font-size: 0.85rem;">
                Click a node to see details
            </div>
        </div>
    </div>

    <div id="graph"></div>

    <script>
        const nodes = {nodes_json};
        const edges = {edges_json};
        const clusters = {clusters_json};
        const gaps = {gaps_json};
        const stats = {stats_json};

        // Update stats display
        document.getElementById('stat-docs').textContent = stats.total_documents;
        document.getElementById('stat-chunks').textContent = stats.total_chunks;
        document.getElementById('stat-clusters').textContent = stats.num_clusters;
        document.getElementById('stat-gaps').textContent = stats.num_gaps;
        document.getElementById('coverage-bar').style.width = (stats.coverage_score * 100) + '%';

        // Render clusters list
        const clustersList = document.getElementById('clusters-list');
        clusters.forEach(cluster => {{
            const div = document.createElement('div');
            div.className = 'cluster-item';
            div.innerHTML = `
                <div class="cluster-dot" style="background: ${{cluster.color}}"></div>
                <div class="cluster-info">
                    <div class="cluster-name">${{cluster.label}}</div>
                    <div class="cluster-count">${{cluster.chunk_count}} chunks</div>
                </div>
            `;
            div.onclick = () => focusCluster(cluster.cluster_id);
            clustersList.appendChild(div);
        }});

        // Render gaps list
        const gapsList = document.getElementById('gaps-list');
        if (gaps.length === 0) {{
            gapsList.innerHTML = '<div style="color: #3fb950; font-size: 0.85rem;">No gaps detected</div>';
        }} else {{
            gaps.forEach(gap => {{
                const div = document.createElement('div');
                div.className = 'gap-item';
                div.innerHTML = `<span class="gap-icon">&#9888;</span>${{gap.description}} (${{gap.isolated_chunk_ids.length}} chunks)`;
                div.onclick = () => focusGap(gap);
                div.style.cursor = 'pointer';
                gapsList.appendChild(div);
            }});
        }}

        // Initialize network
        const container = document.getElementById('graph');
        const data = {{ nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) }};
        const options = {{
            nodes: {{ font: {{ face: 'system-ui, sans-serif' }} }},
            edges: {{ smooth: {{ type: 'continuous' }} }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08,
                    damping: 0.4
                }},
                stabilization: {{ iterations: 150 }}
            }},
            interaction: {{ hover: true, tooltipDelay: 200 }}
        }};

        const network = new vis.Network(container, data, options);

        // Node click handler
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                showNodeDetail(params.nodes[0]);
            }} else {{
                hideNodeDetail();
            }}
        }});

        function showNodeDetail(nodeId) {{
            const node = nodes.find(n => n.id === nodeId);
            if (!node) return;

            document.getElementById('detail-panel').classList.add('active');
            document.getElementById('detail-placeholder').style.display = 'none';

            if (node.nodeType === 'document') {{
                document.getElementById('detail-title').textContent = node.title || node.label;
                document.getElementById('detail-meta').textContent = 'Document';
                document.getElementById('detail-content').textContent =
                    'Contains ' + nodes.filter(n => n.doc_id === node.title).length + ' chunks';
            }} else {{
                document.getElementById('detail-title').textContent = node.label;
                document.getElementById('detail-meta').textContent =
                    'From: ' + node.doc_id + (node.is_gap ? ' (Gap)' : '');
                document.getElementById('detail-content').textContent =
                    node.content_preview || 'No preview available';
            }}
        }}

        function hideNodeDetail() {{
            document.getElementById('detail-panel').classList.remove('active');
            document.getElementById('detail-placeholder').style.display = 'block';
        }}

        function focusCluster(clusterId) {{
            const clusterNodes = nodes.filter(n => n.cluster_id === clusterId).map(n => n.id);
            network.selectNodes(clusterNodes);
            network.fit({{ nodes: clusterNodes, animation: true }});
        }}

        function focusGap(gap) {{
            network.selectNodes(gap.isolated_chunk_ids);
            network.fit({{ nodes: gap.isolated_chunk_ids, animation: true }});
        }}
    </script>
</body>
</html>
"""


__all__ = ["generate_html"]
