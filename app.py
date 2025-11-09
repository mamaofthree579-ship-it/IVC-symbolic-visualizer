import streamlit as st
import pandas as pd

from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
    compute_symbol_energy,
    compute_energy_flow
)

from src.vector_plot import (
    render_3d_resonance_field,
    render_energy_flow_field,
    render_frequency_spectrum
)

# --- Page Setup ---
st.set_page_config(page_title="IVC Symbolic Visualizer", layout="wide")
st.title("🌐 IVC Symbolic Energy Visualizer")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Visualization Options")

# Data size selector
n_symbols = st.sidebar.slider("Number of Symbols", 4, 20, 7)
show_data = st.sidebar.checkbox("Show Data Tables", value=True)
visual_choice = st.sidebar.multiselect(
    "Select Visualization(s) to Display",
    ["Resonance Field (3D)", "Energy Flow Field", "Frequency Spectrum"],
    default=["Resonance Field (3D)", "Energy Flow Field"]
)

# --- Data Generation ---
data = generate_sample_data(n_symbols)
matrix = compute_resonance_matrix(data)
clusters = find_resonant_clusters(matrix)
flow_vectors = compute_energy_flow(data)

# --- Optional Data Tables ---
if show_data:
    st.subheader("Sample Symbolic Data")
    st.dataframe(data)

    st.subheader("Resonance Matrix")
    st.dataframe(matrix)

    st.subheader("Resonant Clusters")
    st.write(clusters)

# --- Visualizations ---
if "Resonance Field (3D)" in visual_choice:
    st.subheader("3D Resonance Field")
    fig_res = render_3d_resonance_field(matrix, clusters)
    st.plotly_chart(fig_res, use_container_width=True)

if "Energy Flow Field" in visual_choice:
    st.subheader("Symbolic Energy Flow Field")
    fig_flow = render_energy_flow_field(data, flow_vectors)
    st.plotly_chart(fig_flow, use_container_width=True)

if "Frequency Spectrum" in visual_choice:
    st.subheader("Symbolic Frequency Spectrum")
    fig_freq = render_frequency_spectrum(data)
    st.plotly_chart(fig_freq, use_container_width=True)

st.success("✅ Visualization complete and interactive.")
st.caption("Use the sidebar to adjust data and visualization layers.")
