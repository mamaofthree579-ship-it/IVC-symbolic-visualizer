"""
src/vector_plot.py

3D and 2D visualization helpers for the IVC Symbolic Visualizer project.

Provides:
- plot_vector_field_matplotlib(data): Matplotlib quiver plot (2D).
- plot_vector_field_plotly(data): Plotly quiver-style visualization (2D).
- render_3d_resonance_field(matrix, labels=None, height=600):
    Create an interactive Plotly 3D visualization of a resonance matrix.
    Nodes are positioned via a simple dimensionality reduction (PCA) to 3D;
    edges are drawn as 3D lines with thickness proportional to resonance strength.
"""

from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

# Matplotlib used for legacy 2D displays (optional)
import matplotlib.pyplot as plt

# Plotly for interactive visuals in Streamlit
import plotly.graph_objects as go

# PCA for positioning nodes in 3D space
from sklearn.decomposition import PCA


def _ensure_matrix(matrix: np.ndarray) -> np.ndarray:
    """Ensure the input is a square numpy array."""
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
        return mat
    raise ValueError("matrix must be a square 2D array (NxN)")


def plot_vector_field_matplotlib(data: np.ndarray, figsize: Tuple[int, int] = (6, 6)):
    """
    Draw a 2D quiver plot using matplotlib.
    :param data: numpy array with shape (N, 4) representing [x, y, u, v].
    :return: matplotlib.figure.Figure instance.
    """
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("data must be shape (N, 4) or (M, N, 4)")

    x, y, u, v = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    fig, ax = plt.subplots(figsize=figsize)
    ax.quiver(x, y, u, v)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Vector Field (matplotlib)')
    plt.tight_layout()
    return fig


def plot_vector_field_plotly(data: np.ndarray, scale: float = 1.0, width: int = 800, height: int = 600):
    """
    Create a 2D quiver-like plot using Plotly.
    :param data: numpy array shape (N, 4) containing x,y,u,v columns.
    :param scale: visual arrow length scaling factor.
    :return: plotly.graph_objects.Figure
    """
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError("data must be shape (N, 4)")
    x, y, u, v = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    # Normalize for consistent arrowhead sizing
    mag = np.sqrt(u * u + v * v)
    mag_max = max(mag.max(), 1e-6)
    u_n = u / mag_max * scale
    v_n = v / mag_max * scale

    fig = go.Figure()

    # Add lines for arrows (from (x,y) to (x+u_n, y+v_n))
    for xi, yi, ui, vi, m in zip(x, y, u_n, v_n, mag):
        fig.add_trace(go.Scatter(
            x=[xi, xi + ui],
            y=[yi, yi + vi],
            mode="lines",
            line=dict(color='royalblue', width=2),
            hoverinfo='text',
            text=f"mag={m:.3f}",
            showlegend=False
        ))
        # small marker at arrow tip
        fig.add_trace(go.Scatter(
            x=[xi + ui],
            y=[yi + vi],
            mode="markers",
            marker=dict(size=4, color='royalblue'),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(title="Vector Field (plotly)", xaxis_title="X", yaxis_title="Y",
                      width=width, height=height, template="simple_white")
    return fig


def render_3d_resonance_field(matrix: np.ndarray,
                              labels: Optional[Sequence[str]] = None,
                              min_edge_threshold: float = 0.2,
                              height: int = 700,
                              node_size: int = 6) -> go.Figure:
    """
    Render a resonance matrix as an interactive 3D Plotly figure.
    - Positions are computed via PCA to 3D from the matrix rows.
    - Edges are drawn between nodes where resonance >= min_edge_threshold.
    - Edge thickness and opacity scale with resonance strength.

    :param matrix: square NxN numpy array or pandas DataFrame
    :param labels: optional list of N labels for nodes
    :param min_edge_threshold: float cutoff (0..1) for showing edges
    :param height: plot height in pixels
    :param node_size: base marker size for nodes
    :return: plotly.graph_objects.Figure
    """
    # Accept pandas DataFrame as input
    if isinstance(matrix, pd.DataFrame):
        labels = list(matrix.index) if labels is None else labels
        mat = matrix.values.astype(float)
    else:
        mat = np.asarray(matrix, dtype=float)
    mat = _ensure_matrix(mat)
    n = mat.shape[0]

    if labels is None:
        labels = [f"Node_{i}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match matrix size")

    # Use PCA on the matrix rows (or columns) to get 3D coordinates
    try:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(mat)
    except Exception:
        # Fallback: random layout but deterministic
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((n, 3))

    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = go.Figure()

    # Add edges as 3D lines
    for i in range(n):
        for j in range(i + 1, n):
            strength = float(mat[i, j])
            if strength >= min_edge_threshold:
                x0, y0, z0 = xs[i], ys[i], zs[i]
                x1, y1, z1 = xs[j], ys[j], zs[j]
                # Scale line width and opacity by strength
                width = max(1.0, 6.0 * strength)
                opacity = min(0.95, 0.25 + 0.75 * strength)
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(color='gray', width=width),
                    opacity=opacity,
                    hoverinfo='none',
                    showlegend=False
                ))

    # Add node points
    node_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(
            size=node_size,
            color=np.linspace(0, 1, n),
            colorscale='Viridis',
            line=dict(width=1, color='DarkSlateGrey')
        ),
        hovertext=[f"{labels[i]}<br>avg_resonance={mat[i].mean():.3f}" for i in range(n)],
        hoverinfo='text',
        showlegend=False
    )
    fig.add_trace(node_trace)

    fig.update_layout(
        title="3D Resonance Field",
        width=1000,
        height=height,
        scene=dict(
            xaxis=dict(title='X', showbackground=False),
            yaxis=dict(title='Y', showbackground=False),
            zaxis=dict(title='Z', showbackground=False),
            aspectmode='auto'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig
