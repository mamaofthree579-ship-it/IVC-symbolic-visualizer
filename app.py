import streamlit as st
import pandas as pd
import numpy as np

# Import your internal modules
from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
    compute_energy_flow,
    compute_symbol_energy
)

from src.vector_plot import render_3d_resonance_field, render_energy_flow_field

# ---------------------------------------------------------------------
# Page Setup
# ---------------------------------------------------------------------
st.set_page_config(page_title="IVC Symbolic Visualizer", layout="wide")
st.title("IVC Symbolic Resonance & Energy Field Visualizer")

st.markdown(
    """
    This tool visualizes intersymbolic resonance, energy flow, and dynamic field
    interactions derived from the IVC decoding framework.
    Use the controls below to adjust matrix size and flow sensitivity.
    """
)

# ---------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------
n_symbols = st.sidebar.slider("Number of Symbols", 4, 20, 8)
flow_strength = st.sidebar.slider("Energy Flow Strength", 0.1, 2.0, 1.0)
show_energy = st.sidebar.checkbox("Show Symbol Energy Mapping", True)

# ---------------------------------------------------------------------
# Data Generation & Analytics
# ---------------------------------------------------------------------
st.subheader("Generating Symbolic Resonance Data...")

data = generate_sample_data(n_symbols)
st.dataframe(data.style.background_gradient(cmap="Blues"), use_container_width=True)

matrix = compute_resonance_matrix(data)
clusters = find_resonant_clusters(matrix)
flow_vectors = compute_energy_flow(data)
symbol_energy = compute_symbol_energy(data)

# ---------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------
st.subheader("3D Resonance Field Visualization")
fig_resonance = render_3d_resonance_field(matrix, clusters)
st.plotly_chart(fig_resonance, use_container_width=True)

st.subheader("3D Symbolic Energy Flow Field")
fig_flow = render_energy_flow_field(data, flow_vectors)
st.plotly_chart(fig_flow, use_container_width=True)

# ---------------------------------------------------------------------
# Optional Energy Mapping
# ---------------------------------------------------------------------
if show_energy:
    st.subheader("Symbolic Energy Mapping")
    energy_df = pd.DataFrame(symbol_energy, columns=["Energy"])
    st.bar_chart(energy_df)

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    """
    ---
    **IVC Symbolic Visualizer** | Exploring intersymbolic resonance, energy flow,
    and dynamic field coherence through geometric and frequency-based models.
    """
)
