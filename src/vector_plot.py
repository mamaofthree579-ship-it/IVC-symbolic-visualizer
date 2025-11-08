import plotly.graph_objects as go
import numpy as np
import streamlit as st

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
