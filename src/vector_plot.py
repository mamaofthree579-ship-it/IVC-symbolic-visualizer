import plotly.graph_objects as go
import numpy as np

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
