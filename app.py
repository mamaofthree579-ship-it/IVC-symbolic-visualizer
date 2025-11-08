import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
from pathlib import Path
import sys

st.set_page_config(page_title="Symbolic Resonance Visualizer", layout="wide")
st.title("🌐 Symbolic Resonance Visualizer")

# --- Load modules
BASE_DIR = Path(__file__).resolve().parent
MODULES_DIR = BASE_DIR / "modules"
SRC_DIR = BASE_DIR / "src"

for p in [MODULES_DIR, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
)

# --- Sidebar controls
st.sidebar.header("Configuration")
num_symbols = st.sidebar.slider("Number of symbols", 3, 15, 6)
threshold = st.sidebar.slider("Resonance threshold", 0.0, 1.0, 0.8)

# --- Generate data
data = generate_sample_data(num_symbols)
matrix = compute_resonance_matrix(data)
clusters = find_resonant_clusters(matrix, threshold)

st.subheader("📊 Resonance Matrix")
st.dataframe(matrix.style.background_gradient(cmap="viridis"))

# --- Heatmap visualization
fig = go.Figure(
    data=go.Heatmap(
        z=matrix.values,
        x=matrix.columns,
        y=matrix.index,
        colorscale="Viridis",
        zmin=0,
        zmax=1,
    )
)
fig.update_layout(title="Resonance Heatmap", height=500)
st.plotly_chart(fig, use_container_width=True)

# --- Build network graph
st.subheader("🌐 Resonant Cluster Network")

G = nx.Graph()

# Add nodes and edges based on resonance threshold
for i in matrix.index:
    G.add_node(i)
for i in matrix.index:
    for j in matrix.columns:
        if i != j and matrix.loc[i, j] >= threshold:
            G.add_edge(i, j, weight=matrix.loc[i, j])

pos = nx.spring_layout(G, seed=42)

edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    line=dict(width=0.5, color="#888"),
    hoverinfo="none",
    mode="lines",
)

node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_text.append(node)

node_trace = go.Scatter(
    x=node_x,
    y=node_y,
    mode="markers+text",
    text=node_text,
    textposition="bottom center",
    hoverinfo="text",
    marker=dict(
        showscale=True,
        colorscale="YlGnBu",
        size=15,
        color=np.linspace(0, 1, len(G.nodes())),
        colorbar=dict(thickness=15, title="Node Index", xanchor="left"),
    ),
)

fig2 = go.Figure(data=[edge_trace, node_trace])
fig2.update_layout(
    showlegend=False,
    title="Symbolic Resonance Network",
    hovermode="closest",
    margin=dict(b=0, l=0, r=0, t=40),
)
st.plotly_chart(fig2, use_container_width=True)

# --- Show clusters
st.subheader("🔹 Identified Clusters")
for c in clusters:
    st.write(", ".join(c))

st.success("✅ Visualization complete.")
