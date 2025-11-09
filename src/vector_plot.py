import numpy as np
import plotly.graph_objects as go

def render_3d_resonance_field(matrix, clusters):
    """3D point cloud for resonance relationships."""
    n = matrix.shape[0]
    np.random.seed(42)
    positions = np.random.rand(n, 3) * 10
    x, y, z = positions.T

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        marker=dict(size=6, color=np.mean(matrix.to_numpy(), axis=1), colorscale="Viridis"),
        text=list(matrix.index),
        textposition="top center"
    ))

    for cluster in clusters:
        if len(cluster) > 1:
            pts = [matrix.index.get_loc(name) for name in cluster]
            for i in pts:
                for j in pts:
                    if i < j:
                        fig.add_trace(go.Scatter3d(
                            x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                            mode="lines", line=dict(color="lightgray", width=2)
                        ))

    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    return fig


def render_energy_flow_field(df, flow_vectors):
    """3D quiver-like visualization of symbolic flow."""
    n = df.shape[0]
    np.random.seed(0)
    positions = np.random.rand(n, 3) * 10
    x, y, z = positions.T
    u, v, w = flow_vectors.mean(axis=1), flow_vectors.std(axis=1), np.gradient(flow_vectors.mean(axis=1))

    fig = go.Figure(data=[
        go.Cone(
            x=x, y=y, z=z,
            u=u, v=v, w=w,
            colorscale="Plasma",
            sizemode="scaled",
            sizeref=2,
            showscale=True
        )
    ])
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    return fig


def render_frequency_spectrum(df):
    """Symbolic energy spectrum as bar chart."""
    energies = np.linalg.norm(df.to_numpy(), axis=1)
    fig = go.Figure(go.Bar(
        x=df.index,
        y=energies,
        marker_color="mediumturquoise"
    ))
    fig.update_layout(
        xaxis_title="Symbol",
        yaxis_title="Energy Level",
        title="Symbolic Energy Spectrum"
    )
    return fig
