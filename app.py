import streamlit as st
import pandas as pd
from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
    compute_symbol_energy,
    compute_energy_flow
)
from src.vector_plot import render_3d_resonance_field, render_energy_flow_field

# --- Streamlit app setup ---
st.set_page_config(page_title="IVC Symbolic Visualizer", layout="wide")

st.title("🔮 IVC Symbolic Visualizer")
st.markdown("Visualize symbolic resonance and energetic relationships between symbols.")

# --- Generate / Load data ---
st.sidebar.header("Data Options")
num_symbols = st.sidebar.slider("Number of Symbols", 3, 12, 6)
data = generate_sample_data(num_symbols)

st.subheader("Generated Symbol Data")
st.dataframe(data)

# --- Resonance matrix computation ---
matrix = compute_resonance_matrix(data)
clusters = find_resonant_clusters(matrix)

st.subheader("Resonance Matrix")
st.dataframe(matrix.style.background_gradient(cmap="PuBuGn"))

# --- 3D Resonance Visualization ---
st.subheader("3D Resonance Field")
fig_resonance = render_3d_resonance_field(matrix, clusters)
st.plotly_chart(fig_resonance, use_container_width=True)

# --- Energy Field Visualization ---
st.subheader("Symbolic Energy Flow Field")
symbol_energy = compute_symbol_energy(data)
flow_vectors = compute_energy_flow(data)
fig_flow = render_energy_flow_field(data, flow_vectors)
st.plotly_chart(fig_flow, use_container_width=True)

st.success("Visualization complete ✅")
