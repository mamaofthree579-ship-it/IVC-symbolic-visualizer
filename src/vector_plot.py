import numpy as np
import plotly.graph_objects as go
import streamlit as st

# --- Energy Flow Field (same as before) ---
def render_energy_flow_field(df, flow_vectors):
    try:
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
        base_mag = np.linalg.norm(flow_vectors, axis=1)
        u, v, w = u / (base_mag + 1e-9), v / (base_mag + 1e-9), w / (base_mag + 1e-9)

        frames = []
        t_values = np.linspace(0, 2 * np.pi, 30)
        for t in t_values:
            scale = 0.5 + 0.5 * np.sin(t + np.arange(n))
            u_t, v_t, w_t = u * scale, v * scale, w * scale
            color_t = base_mag * scale
            frame = go.Frame(
                data=[
                    go.Cone(x=x, y=y, z=z, u=u_t, v=v_t, w=w_t, colorscale="Turbo", sizemode="scaled", sizeref=0.6),
                    go.Scatter3d(x=x, y=y, z=z, mode="markers+text", text=symbols,
                                 marker=dict(size=6, color=color_t, colorscale="Viridis", opacity=0.9))
                ],
                name=f"frame_{t:.2f}"
            )
            frames.append(frame)

        fig = go.Figure(
            data=[
                go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, colorscale="Turbo", sizemode="scaled", sizeref=0.6),
                go.Scatter3d(x=x, y=y, z=z, mode="markers+text", text=symbols,
                             marker=dict(size=6, color=base_mag, colorscale="Viridis", opacity=0.9))
            ],
            layout=go.Layout(
                title="🌐 Animated Symbolic Energy Flow Field",
                scene=dict(aspectmode="cube"),
                margin=dict(l=0, r=0, b=0, t=50),
                updatemenus=[{
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
                         "label": "▶️ Play", "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                         "label": "⏸️ Pause", "method": "animate"}
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "showactive": True,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            ),
            frames=frames
        )
        return fig

    except Exception as e:
        st.error(f"Energy flow visualization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return go.Figure()


# --- NEW: Symbolic Frequency Spectrum ---
def render_frequency_spectrum(df, flow_vectors):
    try:
        n = len(flow_vectors)
        symbols = df.columns if hasattr(df, "columns") else [f"Symbol_{i}" for i in range(n)]

        # Simulate frequency over time for each symbol
        time = np.linspace(0, 10, 300)
        base_freqs = np.linspace(0.5, 2.5, n)
        amplitudes = np.linalg.norm(flow_vectors, axis=1)
        colors = np.linspace(0, 1, n)

        fig = go.Figure()

        for i, sym in enumerate(symbols):
            y = amplitudes[i] * np.sin(2 * np.pi * base_freqs[i] * time)
            fig.add_trace(go.Scatter(
                x=time, y=y,
                mode="lines",
                line=dict(width=2),
                name=sym
            ))

        fig.update_layout(
            title="Frequency Spectrum of Symbolic Energy",
            xaxis_title="Time",
            yaxis_title="Amplitude",
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=30),
            legend_title="Symbols"
        )
        return fig

    except Exception as e:
        st.error(f"Frequency spectrum visualization failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return go.Figure()
