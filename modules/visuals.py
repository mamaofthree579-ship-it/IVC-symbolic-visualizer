# modules/visuals.py
"""
Visualization utilities for the IVC Symbolic Visualizer.
This module provides heatmaps, network graphs, and basic
symbolic vector field representations.
"""

import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# --- 1. Resonance Matrix Visualization ---
def render_resonance_heatmap(matrix, labels=None, title="Resonance Matrix"):
    """
    Display a heatmap of the resonance matrix.
    """
    if matrix is None:
        st.warning("No matrix data available for visualization.")
        return

    # Ensure numpy array
    mat = np.array(matrix, dtype=float)
    n = mat.shape[0]

    if labels is None:
        labels = [f"Node_{i}" for i in range(n)]

    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale="Viridis",
            hoverongaps=False,
            colorbar=dict(title="Resonance")
        )
    )
    fig.update_layout(title=title, xaxis_title="Symbols", yaxis_title="Symbols")
    st.plotly_chart(fig, use_container_width=True)


# --- 2. Cluster Graph Visualization ---
def render_resonant_clusters(matrix, clusters, labels=None, title="Resonant Clusters"):
    """
    Visualize clusters of resonant symbols as a network graph.
    """
    if matrix is None or clusters is None:
        st.warning("No cluster data available for visualization.")
        return

    mat = np.array(matrix, dtype=float)
    G = nx.Graph()

    if labels is None:
        labels = [f"Node_{i}" for i in range(mat.shape[0])]

    # Add edges for each cluster
    for group in clusters:
        for i in group:
            for j in group:
                if i != j:
                    G.add_edge(labels[i], labels[j], weight=mat[i, j])

    # Draw layout
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="gray",
        node_size=700,
        font_size=10
    )
    plt.title(title)
    st.pyplot(plt)


# --- 3. Simple Energy Flow Plot ---
def render_energy_field(vectors, title="Energy Flow Field"):
    """
    Display a quiver plot showing energy flow (2D vector field).
    """
    if vectors is None or len(vectors) == 0:
        st.info("No vector data provided for energy field visualization.")
        return

    vectors = np.array(vectors)
    X, Y = np.meshgrid(np.linspace(-1, 1, vectors.shape[0]), np.linspace(-1, 1, vectors.shape[1]))
    U = np.sin(np.pi * X) * np.cos(np.pi * Y)
    V = -np.cos(np.pi * X) * np.sin(np.pi * Y)

    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V, color="teal")
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    st.pyplot(plt)


# --- 4. Dispatcher Function ---
def render_symbol_map(matrix=None, clusters=None, vectors=None):
    """
    A unified renderer to visualize matrix, clusters, and field data in sequence.
    """
    st.subheader("Visual Overview")

    if matrix is not None:
        render_resonance_heatmap(matrix)

    if clusters is not None and len(clusters) > 0:
        render_resonant_clusters(matrix, clusters)

    if vectors is not None:
        render_energy_field(vectors)
