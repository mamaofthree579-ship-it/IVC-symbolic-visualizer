import streamlit as st
import numpy as np
import json
from modules.visualization import (
    plot_vector_field,
    plot_symbolic_lattice,
    show_frequency_chart
)
from modules.mapping import (
    load_lattice,
    generate_connections,
    merge_lattices,
    build_symbolic_lattice_from_vector_data
)
from modules.analytics import (
    calculate_symbol_frequencies,
    resonance_matrix,
    find_resonant_clusters,
    convert_matrix_to_edges,
    generate_resonance_spectrum
)

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="IVC Symbolic Visualizer",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🌀 IVC Symbolic Visualization System")
st.markdown("A live symbolic and energetic mapping interface based on vector and lattice dynamics.")

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
view = st.sidebar.radio(
    "Select View:",
    ["Vector Field", "Symbolic Lattice", "Frequency Analysis"]
)

# --- SYNTHETIC SAMPLE DATA (auto-generates if none provided) ---
def generate_sample_vector_data(n=12):
    x = np.linspace(0, 2*np.pi, n)
    y = np.linspace(0, 2*np.pi, n)
    X, Y = np.meshgrid(x, y)
    U = np.cos(X) * np.sin(Y)
    V = np.sin(X) * np.cos(Y)
    data = np.column_stack([X.flatten(), Y.flatten(), U.flatten(), V.flatten()])
    return data

# --- VIEW 1: VECTOR FIELD VISUALIZATION ---
if view == "Vector Field":
    st.subheader("Vector Flow Map")
    st.write("Visualizes symbolic energy or directional flow patterns.")

    n_points = st.slider("Number of sample points", 6, 30, 12)
    data = generate_sample_vector_data(n_points)
    plot_vector_field(data, title="Sample Energy Vector Field")

# --- VIEW 2: SYMBOLIC LATTICE VISUALIZATION ---
elif view == "Symbolic Lattice":
    st.subheader("Symbolic Lattice Network")
    st.write("Displays relational connections between symbolic nodes.")

    mode = st.radio("Connection Pattern", ["pairwise", "complete", "random"])
    nodes = ["Water", "Earth", "Air", "Fire", "Spirit"]
    connections = generate_connections(nodes, pattern=mode)
    st.json({"nodes": nodes, "connections": connections})
    plot_symbolic_lattice(connections, title=f"{mode.capitalize()} Symbolic Network")

# --- VIEW 3: FREQUENCY / RESONANCE ANALYSIS ---
elif view == "Frequency Analysis":
    st.subheader("Frequency and Resonance Analysis")
    st.write("Computes resonance and harmonic clustering between symbolic elements.")

    data = generate_sample_vector_data(10)
    matrix = resonance_matrix(data)
    clusters = find_resonant_clusters(matrix)

    st.markdown("**Resonant Clusters:**")
    st.write(clusters)

    labels = [f"Node_{i}" for i in range(matrix.shape[0])]
    spectrum = generate_resonance_spectrum(matrix, labels)
    show_frequency_chart(spectrum, title="Symbolic Resonance Spectrum")

    st.markdown("**Matrix Preview:**")
    st.dataframe(matrix)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for symbolic-energetic research and visualization.")
