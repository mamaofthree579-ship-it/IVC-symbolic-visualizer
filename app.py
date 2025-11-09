import streamlit as st
import numpy as np
import pandas as pd

from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters
)

from src.vector_plot import (
    render_3d_resonance_field,
    render_energy_flow_field,
    render_frequency_spectrum
)

st.set_page_config(page_title="IVC Energy Visualizer", layout="wide")

st.title("🌐 IVC Symbolic Energy & Resonance Visualizer")

# --- Load or generate data ---
st.sidebar.header("Dataset Controls")
num_symbols = st.sidebar.slider("Number of symbols", 3, 15, 6)
data = generate_sample_data(num_symbols)
matrix = compute_resonance_matrix(data)
clusters = find_resonant_clusters(matrix)

# --- Generate flow vectors ---
np.random.seed(42)
flow_vectors = np.random.randn(num_symbols, 3)

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("3D Resonance Energy Field")
    fig_field = render_energy_flow_field(data, flow_vectors)
    st.plotly_chart(fig_field, use_container_width=True)

with col2:
    st.subheader("Symbolic Frequency Spectrum")
    freq_fig = render_frequency_spectrum(data, flow_vectors)
    st.plotly_chart(freq_fig, use_container_width=True)
