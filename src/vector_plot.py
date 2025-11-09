import plotly.graph_objects as go
import numpy as np
import streamlit as st
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
    Render a 3D energy flow field visualization for symbolic energy dynamics.
    df: DataFrame with symbols
    flow_vectors: (n,3) array of energy flow components
    """
    symbols = df.columns
    n = len(symbols)

    # Ensure flow_vectors shape matches symbols
    if flow_vectors.shape[0] != n:
        flow_vectors = np.resize(flow_vectors, (n, 3))

    # Create positions in 3D space
    theta = np.linspace(0, 2*np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.linspace(-1, 1, n)

    # Compute end points for flow vectors
    u, v, w = flow_vectors[:, 0], flow_vectors[:, 1], flow_vectors[:, 2]

    # Normalize for visual scaling
    mag = np.linalg.norm(flow_vectors, axis=1)
    u, v, w = u / (mag + 1e-9), v / (mag + 1e-9), w / (mag + 1e-9)

    # Build 3D cone plot
    fig = go.Figure(
        data=go.Cone(
            x=x, y=y, z=z,
            u=u, v=v, w=w,
            colorscale="Turbo",
            sizemode="scaled",
            sizeref=0.6,
            showscale=True,
            colorbar_title="Energy Flow"
        )
    )

    # Overlay points for symbols
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        text=symbols,
        textposition="top center",
        marker=dict(size=5, color=mag, colorscale="Viridis", opacity=0.8),
        name="Symbols"
    ))

    # Layout aesthetics
    fig.update_layout(
        title="3D Symbolic Energy Flow Field",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig
    
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
    
def render_energy_flow_field(df, flow_vectors):
    """
    Render a 3D energy flow field visualization for symbolic energy dynamics.
    df: DataFrame with symbols
    flow_vectors: (n,3) array of energy flow components
    """
    try:
        st.write("DataFrame shape:", df.shape)
        st.write("Flow vectors shape:", getattr(flow_vectors, "shape", "None"))

        symbols = df.columns if hasattr(df, "columns") else [f"Symbol_{i}" for i in range(len(flow_vectors))]
        n = len(symbols)

        # Ensure flow_vectors is a numpy array
        flow_vectors = np.array(flow_vectors)
        if flow_vectors.ndim == 1:
            flow_vectors = np.expand_dims(flow_vectors, axis=1)

        # Resize safely if mismatched
        if flow_vectors.shape[0] != n:
            st.warning(f"Resizing flow_vectors from {flow_vectors.shape} to ({n}, 3)")
            flow_vectors = np.resize(flow_vectors, (n, 3))

        # Create positions
        theta = np.linspace(0, 2*np.pi, n)
        x = np.cos(theta)
        y = np.sin(theta)
        z = np.linspace(-1, 1, n)

        # Components
        u, v, w = flow_vectors[:, 0], flow_vectors[:, 1], flow_vectors[:, 2]
        mag = np.linalg.norm(flow_vectors, axis=1)
        u, v, w = u / (mag + 1e-9), v / (mag + 1e-9), w / (mag + 1e-9)

        fig = go.Figure()

        fig.add_trace(go.Cone(
            x=x, y=y, z=z,
            u=u, v=v, w=w,
            colorscale="Turbo",
            sizemode="scaled",
            sizeref=0.6,
            showscale=True,
            colorbar_title="Energy Flow"
        ))

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            text=symbols,
            textposition="top center",
            marker=dict(size=5, color=mag, colorscale="Viridis", opacity=0.8),
            name="Symbols"
        ))

        fig.update_layout(
            title="3D Symbolic Energy Flow Field",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="cube"
            ),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        return fig

    except Exception as e:
        st.error(f"Energy flow visualization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return go.Figure()
