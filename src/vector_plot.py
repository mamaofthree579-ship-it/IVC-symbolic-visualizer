import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- Resonance Field ---
def render_3d_resonance_field(matrix, clusters):
    symbols = matrix.index.tolist()
    n = len(symbols)
    theta = np.linspace(0, 2 * np.pi, n)
    x = np.cos(theta)
    y = np.sin(theta)
    z = matrix.mean(axis=1).values

    fig = go.Figure()

    # Draw resonance connections
    for i in range(n):
        for j in range(i + 1, n):
            val = matrix.iloc[i, j]
            if val > 0.75:
                fig.add_trace(go.Scatter3d(
                    x=[x[i], x[j]], y=[y[i], y[j]], z=[z[i], z[j]],
                    mode="lines",
                    line=dict(width=2, color=f"rgba(0,100,255,{val})"),
                    showlegend=False
                ))

    # Draw symbols
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        text=symbols,
        textposition="top center",
        marker=dict(size=6, color=z, colorscale="Viridis", opacity=0.8),
        name="Symbols"
    ))

    fig.update_layout(
        title="3D Resonance Field",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    return fig


# --- Energy Flow Field ---
def render_energy_flow_field(df, flow_vectors):
    try:
        st.write("DEBUG: df shape", df.shape)
        st.write("DEBUG: flow_vectors shape", getattr(flow_vectors, "shape", None))

        symbols = df.columns if hasattr(df, "columns") else [f"Symbol_{i}" for i in range(len(flow_vectors))]
        n = len(symbols)

        flow_vectors = np.array(flow_vectors)
        if flow_vectors.shape[0] != n:
            flow_vectors = np.resize(flow_vectors, (n, 3))

        theta = np.linspace(0, 2 * np.pi, n)
        x = np.cos(theta)
        y = np.sin(theta)
        z = np.linspace(-1, 1, n)

        u, v, w = flow_vectors[:, 0], flow_vectors[:, 1], flow_vectors[:, 2]
        mag = np.linalg.norm(flow_vectors, axis=1)
        u, v, w = u / (mag + 1e-9), v / (mag + 1e-9), w / (mag + 1e-9)

        fig = go.Figure()

        fig.add_trace(go.Cone(
            x=x, y=y, z=z,
            u=u, v=v, w=w,
            colorscale="Turbo",
            sizemode="scaled",
            sizeref=0.5,
            showscale=True,
            colorbar_title="Energy Flow"
        ))

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            text=symbols,
            textposition="top center",
            marker=dict(size=5, color=mag, colorscale="Viridis", opacity=0.9),
            name="Symbols"
        ))

        fig.update_layout(
            title="3D Symbolic Energy Flow Field",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="cube"),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        return fig

    except Exception as e:
        st.error(f"Energy flow visualization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return go.Figure()
