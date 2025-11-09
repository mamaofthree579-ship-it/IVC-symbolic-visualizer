# src/vector_plot.py
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import networkx as nx

# 3D resonance field (uses PCA to place nodes)
def render_3d_resonance_field(matrix, clusters):
    if matrix is None or matrix.empty:
        return go.Figure()

    labels = list(matrix.index)
    pca = PCA(n_components=3)
    coords = pca.fit_transform(matrix.values)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = go.Figure()
    # node scatter
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        text=labels,
        textposition="top center",
        marker=dict(size=8, color=np.mean(matrix.values, axis=1), colorscale='Viridis', opacity=0.9)
    ))

    # cluster lines (optional)
    for cluster in clusters:
        indices = [labels.index(s) for s in cluster if s in labels]
        if len(indices) <= 1:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                fig.add_trace(go.Scatter3d(
                    x=[x[indices[i]], x[indices[j]]],
                    y=[y[indices[i]], y[indices[j]]],
                    z=[z[indices[i]], z[indices[j]]],
                    mode='lines',
                    line=dict(color='lightgray', width=2),
                    opacity=0.6,
                    showlegend=False
                ))

    fig.update_layout(title="3D Resonance Field", margin=dict(l=0, r=0, b=0, t=30))
    return fig

# energy flow field rendering using cones
def render_energy_flow_field(df, flow_vectors):
    if df is None:
        return go.Figure()

    labels = list(df.index) if hasattr(df, "index") else [f"Symbol_{i}" for i in range(len(flow_vectors))]
    n = len(labels)

    # ensure array shape
    vectors = np.asarray(flow_vectors)
    if vectors.shape[0] != n:
        vectors = np.resize(vectors, (n, 3))

    # Layout nodes on a circle for visual clarity
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(theta)
    y = np.sin(theta)
    z = np.linspace(-0.5, 0.5, n)

    u, v, w = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    mag = np.linalg.norm(vectors, axis=1)
    # avoid division by zero
    denom = (mag + 1e-9)
    u, v, w = u / denom, v / denom, w / denom

    cone = go.Cone(
        x=x, y=y, z=z,
        u=u, v=v, w=w,
        colorscale='Turbo',
        sizemode='scaled',
        sizeref=0.5,
        showscale=True,
        anchor='tail'
    )

    nodes = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(size=6, color=mag, colorscale='Viridis', opacity=0.9)
    )

    fig = go.Figure(data=[cone, nodes])
    fig.update_layout(title="3D Symbolic Energy Flow Field", margin=dict(l=0, r=0, b=0, t=30))
    return fig

# symbol network visualization (force-directed 3D)
def render_symbolic_network(matrix, threshold=0.6):
    if matrix is None or matrix.empty:
        return go.Figure()

    labels = list(matrix.index)
    G = nx.Graph()
    G.add_nodes_from(labels)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            w = float(matrix.iloc[i, j])
            if w >= threshold:
                G.add_edge(labels[i], labels[j], weight=w)

    if G.number_of_edges() == 0:
        # fallback connect circularly
        for i in range(len(labels)):
            G.add_edge(labels[i], labels[(i + 1) % len(labels)], weight=0.2)

    pos = nx.spring_layout(G, dim=3, seed=42)

    # build edge traces
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(width=2, color='lightblue'))

    # node traces
    node_x, node_y, node_z = [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x); node_y.append(y); node_z.append(z)

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=[n for n in G.nodes()],
        textposition='top center',
        marker=dict(size=8, color=np.linspace(0, 1, len(node_x)), colorscale='Viridis')
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=f"Symbolic Network (threshold={threshold})", margin=dict(l=0, r=0, b=0, t=30))
    return fig
