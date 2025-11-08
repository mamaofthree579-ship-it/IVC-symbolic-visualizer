import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA

def render_3d_resonance_field(matrix, clusters):
    """
    Render a 3D resonance matrix visualization using PCA projection.
    """
    data = matrix.to_numpy()
    pca = PCA(n_components=3)
    coords = pca.fit_transform(data)
    symbols = matrix.index.tolist()

    # Assign cluster colors
    colors = {}
    palette = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, cluster in enumerate(clusters):
        for sym in cluster:
            colors[sym] = palette[i % len(palette)]

    # Scatter plot for resonance field
    scatter = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers+text",
        text=symbols,
        textposition="top center",
        marker=dict(size=8, color=[colors.get(sym, "#7f7f7f") for sym in symbols]),
    )

    layout = go.Layout(
        title="3D Resonance Field",
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    return go.Figure(data=[scatter], layout=layout)


def render_energy_flow_field(df, flow_vectors):
    """
    Render a 3D energy frequency field showing how energy flows between symbols.
    """
    symbols = list(flow_vectors.keys())
    n = len(symbols)

    # Get coordinates via PCA
    data = df.to_numpy()
    pca = PCA(n_components=3)
    coords = pca.fit_transform(data)

    # Normalize flow vectors
    flow_data = np.array(list(flow_vectors.values()))
    flow_data = flow_data / (np.linalg.norm(flow_data, axis=1, keepdims=True) + 1e-9)

    # Create 3D quiver plot
    quiver = go.Cone(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        u=flow_data[:, 0],
        v=flow_data[:, 1],
        w=flow_data[:, 2],
        sizemode="absolute",
        sizeref=0.5,
        colorscale="Viridis",
        showscale=True,
        name="Energy Flow",
    )

    points = go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        text=symbols,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=5, color="white", opacity=0.8),
        name="Symbols"
    )

    layout = go.Layout(
        title="3D Symbolic Energy Flow Field",
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
        ),
        showlegend=True,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return go.Figure(data=[points, quiver], layout=layout)
