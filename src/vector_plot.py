import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# Utility: Generate consistent 3D coordinates from resonance data
# ------------------------------------------------------------------------------
def _get_3d_coords(matrix):
    """Reduce resonance matrix into 3D coordinates using PCA."""
    try:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(matrix)
        return coords
    except Exception:
        # Fallback if PCA fails
        n = matrix.shape[0]
        return np.random.rand(n, 3)


# ------------------------------------------------------------------------------
# 3D Resonance Field Visualization
# ------------------------------------------------------------------------------
def render_3d_resonance_field(matrix, clusters):
    """Render a 3D field showing symbol resonance and clustering."""
    if matrix is None or matrix.empty:
        raise ValueError("Matrix is empty or None.")

    labels = matrix.index.tolist()
    coords = _get_3d_coords(matrix.values)

    fig = go.Figure()

    # Assign a unique color per cluster
    if clusters is None or len(clusters) == 0:
        clusters = [{label} for label in labels]

    colors = [
        f"hsl({int(360 * i / len(clusters))},70%,60%)"
        for i in range(len(clusters))
    ]

    for i, cluster in enumerate(clusters):
        idxs = [labels.index(label) for label in cluster if label in labels]
        x, y, z = coords[idxs, 0], coords[idxs, 1], coords[idxs, 2]

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                text=[labels[k] for k in idxs],
                marker=dict(size=8, color=colors[i], opacity=0.8),
                name=f"Cluster {i+1}",
            )
        )

    fig.update_layout(
        title="3D Resonance Field of Symbolic Network",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
    )
    return fig


# ------------------------------------------------------------------------------
# Energy Flow Field Visualization
# ------------------------------------------------------------------------------
def render_energy_flow_field(data, flow_vectors):
    """Visualize energy vectors as arrows in 3D space."""
    if data is None or flow_vectors is None:
        raise ValueError("Data or flow vectors not provided.")

    labels = data.index.tolist() if isinstance(data, np.ndarray) is False else [
        f"Symbol_{i}" for i in range(data.shape[0])
    ]

    n = len(labels)
    coords = np.random.rand(n, 3)

    fig = go.Figure()

    for i in range(n):
        x, y, z = coords[i]
        u, v, w = flow_vectors[i]

        fig.add_trace(
            go.Cone(
                x=[x],
                y=[y],
                z=[z],
                u=[u],
                v=[v],
                w=[w],
                colorscale="Viridis",
                sizemode="absolute",
                sizeref=0.4,
                anchor="tail",
                showscale=False,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            text=labels,
            mode="markers+text",
            marker=dict(size=6, color="orange", opacity=0.8),
            name="Symbols",
        )
    )

    fig.update_layout(
        title="Symbolic Energy Flow Field",
        scene=dict(
            xaxis_title="Energy X",
            yaxis_title="Energy Y",
            zaxis_title="Energy Z",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return fig


# ------------------------------------------------------------------------------
# Frequency Spectrum Visualization
# ------------------------------------------------------------------------------
def render_frequency_spectrum(symbol_energy):
    """Render a 3D frequency field or symbolic spectrum."""
    if symbol_energy is None or len(symbol_energy) == 0:
        raise ValueError("Symbol energy data not provided.")

    symbols = list(symbol_energy.keys())
    values = np.array(list(symbol_energy.values()))

    fig = go.Figure(
        data=[
            go.Bar3d(
                x=list(range(len(symbols))),
                y=[0] * len(symbols),
                z=[0] * len(symbols),
                dx=[0.5] * len(symbols),
                dy=[0.5] * len(symbols),
                dz=values,
                text=symbols,
                hovertext=[f"{s}: {v:.2f}" for s, v in zip(symbols, values)],
                hoverinfo="text",
            )
        ]
    )

    fig.update_layout(
        title="Symbolic Frequency Spectrum",
        scene=dict(
            xaxis=dict(title="Symbol Index"),
            yaxis=dict(title="Baseline"),
            zaxis=dict(title="Frequency / Energy"),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
