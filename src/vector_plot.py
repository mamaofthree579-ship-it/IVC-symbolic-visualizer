import numpy as np
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import PCA

# === 3D Resonance Field ===
def render_3d_resonance_field(matrix, clusters):
    """Render 3D resonance relationships among symbols."""
    symbols = matrix.columns
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matrix.values)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = go.Figure()

    # Draw cluster groupings
    for cluster in clusters:
        cluster_indices = [symbols.get_loc(sym) for sym in cluster if sym in symbols]
        fig.add_trace(go.Scatter3d(
            x=x[cluster_indices],
            y=y[cluster_indices],
            z=z[cluster_indices],
            mode='markers+text',
            marker=dict(size=8, opacity=0.8),
            text=[symbols[i] for i in cluster_indices],
            name=f"Cluster {len(cluster)}"
        ))

    fig.update_layout(
        title="3D Resonance Field",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    return fig


# === Energy Flow Visualization ===
def render_energy_flow_field(df, flow_vectors):
    """Visualize symbol energy flow as 3D field."""
    symbols = df.columns
    coords = np.arange(len(symbols))
    X, Y = np.meshgrid(coords, coords)

    U = flow_vectors[0]
    V = flow_vectors[1] if flow_vectors.shape[0] > 1 else np.zeros_like(U)

    fig = go.Figure(data=go.Cone(
        x=X.flatten(),
        y=Y.flatten(),
        z=np.zeros_like(X).flatten(),
        u=U.flatten(),
        v=V.flatten(),
        w=np.zeros_like(U).flatten(),
        colorscale='Viridis',
        sizemode='absolute',
        sizeref=2
    ))

    fig.update_layout(
        title="3D Energy Flow Field",
        scene=dict(xaxis_title="Symbol X", yaxis_title="Symbol Y", zaxis_title="Energy Depth")
    )
    return fig


# === NEW: Symbolic Network Visualization ===
def render_symbolic_network(matrix, threshold=0.6):
    """Interactive 3D symbolic network map."""
    if matrix is None or matrix.empty:
        return go.Figure()

    symbols = list(matrix.columns)
    G = nx.Graph()
    for i in range(len(symbols)):
        G.add_node(symbols[i])
        for j in range(i + 1, len(symbols)):
            weight = float(matrix.iloc[i, j])
            if weight >= threshold:
                G.add_edge(symbols[i], symbols[j], weight=weight)

    if G.number_of_edges() == 0:
        # fallback: connect symbols in a simple loop
        G.add_edges_from([(symbols[i], symbols[(i + 1) % len(symbols)]) for i in range(len(symbols))])

    pos = nx.spring_layout(G, dim=3, seed=42)
    x_nodes, y_nodes, z_nodes = [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        x_nodes.append(x)
        y_nodes.append(y)
        z_nodes.append(z)

    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='lightblue'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        marker=dict(size=8, color='gold', opacity=0.9),
        text=[f"{node}" for node in G.nodes()],
        textposition='top center'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Symbolic Network Connectivity",
        showlegend=False,
        scene=dict(xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        paper_bgcolor='black',
        font=dict(color='white')
    )
    return fig
