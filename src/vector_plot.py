import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# 3D Resonance Field Visualization
# ------------------------------------------------------------------------------
def render_3d_resonance_field(matrix, clusters):
    """Render 3D symbolic resonance field based on matrix similarity."""
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matrix.values)
    symbols = matrix.index.tolist()

    fig = go.Figure()

    # Plot each cluster
    if clusters:
        for cluster in clusters:
            cluster_indices = [symbols.index(s) for s in cluster if s in symbols]
            cluster_coords = coords[cluster_indices]
            fig.add_trace(go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers',
                marker=dict(size=6, color=np.random.rand(), opacity=0.8),
                name=f"Cluster {list(cluster)[0]}"
            ))
    else:
        # fallback single group
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers",
            marker=dict(size=6, color='blue', opacity=0.7),
            name="Resonance Field"
        ))

    fig.update_layout(
        title="3D Resonance Matrix Field",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        height=600
    )
    return fig


# ------------------------------------------------------------------------------
# Symbolic Network Visualization
# ------------------------------------------------------------------------------
def render_symbolic_network(matrix):
    """Simple network graph of symbolic connections based on resonance."""
    import networkx as nx
    G = nx.Graph()
    symbols = matrix.index.tolist()

    for i, s1 in enumerate(symbols):
        for j, s2 in enumerate(symbols):
            if i < j and matrix.iloc[i, j] > 0.7:
                G.add_edge(s1, s2, weight=matrix.iloc[i, j])

    pos = nx.spring_layout(G, seed=42)
    x, y = zip(*[pos[k] for k in G.nodes])

    fig = go.Figure()
    # Nodes
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", text=list(G.nodes),
        textposition="top center", marker=dict(size=12, color="skyblue")
    ))
    # Edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines", line=dict(width=2, color="gray"), hoverinfo="none"
        ))

    fig.update_layout(title="Symbolic Resonance Network", showlegend=False)
    return fig


# ------------------------------------------------------------------------------
# 3D Energy Flow Field
# ------------------------------------------------------------------------------
def render_energy_flow_field(df, flow_vectors):
    """Render 3D vector field showing symbolic energy flow."""
    pca = PCA(n_components=3)
    coords = pca.fit_transform(df.values)
    X, Y, Z = coords[:, 0], coords[:, 1], coords[:, 2]

    U, V, W = flow_vectors[:, 0], flow_vectors[:, 1], flow_vectors[:, 2]

    fig = go.Figure(
        data=go.Cone(
            x=X, y=Y, z=Z, u=U, v=V, w=W,
            colorscale="Viridis", sizemode="absolute", sizeref=2,
            showscale=True
        )
    )
    fig.update_layout(title="Symbolic Energy Flow Field", height=600)
    return fig


# ------------------------------------------------------------------------------
# Frequency Spectrum Visualization (FIXED)
# ------------------------------------------------------------------------------
def render_frequency_spectrum(energy_map):
    """Render a 3D frequency-like spectrum of symbol energy values."""
    if not isinstance(energy_map, dict):
        raise ValueError("energy_map must be a dictionary")

    symbols = list(energy_map.keys())
    energies = np.array(list(energy_map.values()))
    num_symbols = len(symbols)

    # Generate pseudo-frequency and amplitude
    freqs = np.linspace(0.1, 2.0, num_symbols)
    amplitudes = energies / energies.max() * 10.0
    z = amplitudes * np.sin(freqs * np.pi)

    fig = go.Figure()

    # Each bar represented as a vertical line
    for i, sym in enumerate(symbols):
        fig.add_trace(go.Scatter3d(
            x=[freqs[i], freqs[i]],
            y=[0, energies[i]],
            z=[0, z[i]],
            mode="lines+markers",
            line=dict(width=8, color="orange"),
            marker=dict(size=4, color="red"),
            name=sym
        ))

    fig.update_layout(
        title="3D Symbolic Frequency Spectrum",
        scene=dict(
            xaxis_title="Symbol Frequency",
            yaxis_title="Amplitude (Energy)",
            zaxis_title="Resonance Phase"
        ),
        height=600
    )
    return fig
