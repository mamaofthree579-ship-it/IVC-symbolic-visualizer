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

def render_symbolic_network(matrix, threshold=0.6):
    """
    Create an interactive 3D force-directed symbolic network.
    Nodes = symbols, Edges = resonance strength above threshold.
    Auto-adjusts if network would be empty.
    """
    if matrix is None or matrix.empty:
        return go.Figure()

    symbols = list(matrix.columns)
    G = nx.Graph()

    # Add nodes and edges based on resonance strength
    edge_count = 0
    for i in range(len(symbols)):
        G.add_node(symbols[i])
        for j in range(i + 1, len(symbols)):
            try:
                weight = float(matrix.iloc[i, j])
            except:
                continue
            if weight >= threshold:
                G.add_edge(symbols[i], symbols[j], weight=weight)
                edge_count += 1

    # If no edges, lower the threshold gradually
    if edge_count == 0:
        for new_thresh in [0.5, 0.4, 0.3, 0.2]:
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    weight = float(matrix.iloc[i, j])
                    if weight >= new_thresh:
                        G.add_edge(symbols[i], symbols[j], weight=weight)
                        edge_count += 1
            if edge_count > 0:
                threshold = new_thresh
                break

    # If STILL empty, return placeholder layout
    if edge_count == 0:
        G.add_edges_from([(symbols[i], symbols[(i+1) % len(symbols)]) for i in range(len(symbols))])

    pos = nx.spring_layout(G, dim=3, seed=42)
    x_nodes, y_nodes, z_nodes = [], [], []
    for node in G.nodes():
        x, y, z = pos[node]
        x_nodes.append(x)
        y_nodes.append(y)
        z_nodes.append(z)

    edge_x, edge_y, edge_z, edge_colors = [], [], [], []
    for edge in G.edges(data=True):
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_z += [z0, z1, None]
        w = edge[2].get("weight", 0.5)
        edge_colors.append(w)

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color="lightblue"),
        hoverinfo="none",
        mode="lines"
    )

    node_trace = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode="markers+text",
        marker=dict(
            size=8,
            color=np.linspace(0, 1, len(x_nodes)),
            colorscale="Viridis",
            opacity=0.9
        ),
        text=[f"{node}" for node in G.nodes()],
        textposition="top center",
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"🌐 Symbolic Network Connectivity (threshold={threshold})",
        showlegend=False,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="black",
        font=dict(color="white")
    )
    return fig
