import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def render_3d_resonance_field(matrix, clusters=None):
    """
    Render a 3D interactive resonance field with energy mapping and cluster highlighting.
    - matrix: pd.DataFrame, resonance matrix
    - clusters: list of sets of symbols (optional)
    """
    if matrix is None or matrix.empty:
        raise ValueError("Matrix is empty or not provided.")

    symbols = matrix.index.tolist()
    data = matrix.to_numpy()

    # Reduce to 3D for plotting
    pca = PCA(n_components=3)
    coords = pca.fit_transform(data)

    # Compute energy
    energies = matrix.sum(axis=1).to_numpy()
    energy_scaled = (energies - energies.min()) / (energies.max() - energies.min() + 1e-9)

    # Assign cluster colors
    color_map = {}
    palette = [
        "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#E15FED",
        "#F8961E", "#00B4D8", "#9B5DE5", "#F15BB5", "#FF9F1C"
    ]
    if clusters:
        for i, cluster in enumerate(clusters):
            for symbol in cluster:
                color_map[symbol] = palette[i % len(palette)]
    else:
        for symbol in symbols:
            color_map[symbol] = "#6BCB77"

    # 3D scatter for symbols
    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers+text",
        text=symbols,
        textposition="top center",
        marker=dict(
            size=10 + 20 * energy_scaled,
            color=[color_map[sym] for sym in symbols],
            opacity=0.85,
            line=dict(width=1, color="white")
        ),
        hovertemplate="<b>%{text}</b><br>Energy: %{marker.size:.2f}<extra></extra>"
    )

    # Start figure
    fig = go.Figure(data=[scatter])

    # Draw lines between resonant pairs in same cluster
    if clusters:
        for cluster in clusters:
            indices = [symbols.index(s) for s in cluster if s in symbols]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    x0, y0, z0 = coords[indices[i]]
                    x1, y1, z1 = coords[indices[j]]
                    fig.add_trace(go.Scatter3d(
                        x=[x0, x1],
                        y=[y0, y1],
                        z=[z0, z1],
                        mode="lines",
                        line=dict(color=color_map[matrix.index[indices[i]]], width=2),
                        opacity=0.3,
                        showlegend=False
                    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        title="3D Symbolic Resonance Field",
        margin=dict(l=0, r=0, b=0, t=50),
        showlegend=False,
        template="plotly_dark"
    )

    # ✅ Ensure a figure object is always returned
    return fig



def render_energy_flow_field(df, flow_vectors):
    """
    Render a 3D dynamic energy flow visualization.
    Nodes are symbols; lines represent directional energy flow.
    """
    pca = PCA(n_components=3)
    coords = pca.fit_transform(df.values)
    symbol_positions = {sym: coords[i] for i, sym in enumerate(df.index)}

    edge_x, edge_y, edge_z, flow_colors = [], [], [], []

    for (source, target, mag) in flow_vectors:
        x0, y0, z0 = symbol_positions[source]
        x1, y1, z1 = symbol_positions[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
        flow_colors.append(abs(mag))

    node_x, node_y, node_z = coords[:, 0], coords[:, 1], coords[:, 2]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(width=3, color=flow_colors, colorscale='Plasma'),
        opacity=0.8
    )

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=df.index,
        marker=dict(size=10, color='white', opacity=0.9),
        textposition="top center"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="3D Symbolic Energy Flow Field",
        scene=dict(
            xaxis_title="Resonance X",
            yaxis_title="Resonance Y",
            zaxis_title="Resonance Z",
        ),
        showlegend=False,
        height=800
    )

    return fig
