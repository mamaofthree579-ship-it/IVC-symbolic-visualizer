import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_vector_field(data, title="Vector Flow Map"):
    """
    Displays a 2D vector field based on input data.
    :param data: dict or np.ndarray with 'x', 'y', 'u', 'v' components
    :param title: str, title for the plot
    """
    if isinstance(data, dict):
        x, y, u, v = data['x'], data['y'], data['u'], data['v']
    else:
        x, y, u, v = data[:,0], data[:,1], data[:,2], data[:,3]

    fig, ax = plt.subplots(figsize=(6, 6))
    q = ax.quiver(x, y, u, v, color='royalblue', angles='xy', scale_units='xy', scale=1)
    ax.set_title(title)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_aspect('equal')
    st.pyplot(fig)

def plot_symbolic_lattice(connections, title="Symbolic Lattice Map"):
    """
    Draws a symbolic relational map based on node connections.
    :param connections: list of tuples [(node1, node2), ...]
    """
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from(connections)

    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=800, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def show_frequency_chart(frequencies, title="Frequency / Resonance Chart"):
    """
    Visualizes frequencies and their resonance intensity.
    :param frequencies: dict, key=label, value=frequency magnitude
    """
    labels = list(frequencies.keys())
    values = list(frequencies.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color='goldenrod')
    ax.set_title(title)
    ax.set_xlabel('Symbolic Element')
    ax.set_ylabel('Frequency Magnitude (Hz)')
    st.pyplot(fig)
