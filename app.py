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


st.set_page_config(page_title="IVC Symbolic Visualizer", layout="wide")
st.title("IVC Symbolic Energy Visualizer")

# Generate sample data
data = generate_sample_data(7)
st.subheader("Sample Symbolic Data")
st.dataframe(data)

# Resonance matrix
matrix = compute_resonance_matrix(data)
st.subheader("Resonance Matrix")
st.dataframe(matrix)

# Clusters
clusters = find_resonant_clusters(matrix)
st.subheader("Resonant Clusters")
st.write(clusters)

# Resonance 3D field
st.subheader("3D Resonance Field")
fig_res = render_3d_resonance_field(matrix, clusters)
st.plotly_chart(fig_res, use_container_width=True)

# Energy flow
flow_vectors = compute_energy_flow(data)
st.subheader("Symbolic Energy Flow Field")
fig_flow = render_energy_flow_field(data, flow_vectors)
st.plotly_chart(fig_flow, use_container_width=True)

# Frequency spectrum
st.subheader("Symbolic Frequency Spectrum")
fig_freq = render_frequency_spectrum(data)
st.plotly_chart(fig_freq, use_container_width=True)

st.success("Visualization complete and interactive.")
