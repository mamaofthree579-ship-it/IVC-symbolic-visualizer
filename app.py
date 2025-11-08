import streamlit as st
import pandas as pd
import numpy as np

from modules.analytics import generate_sample_data, compute_resonance_matrix, find_resonant_clusters
from src.vector_plot import render_3d_resonance_field

st.set_page_config(page_title="IVC Symbolic Visualizer", layout="wide")

# App title
st.title("🌐 IVC Symbolic Visualizer")
st.write("An interactive system for mapping resonance between symbolic datasets.")

# --- Data Input Section ---
st.sidebar.header("Data Controls")

data_option = st.sidebar.selectbox(
    "Choose input data source:",
    ["Generate Sample Data", "Upload CSV"]
)

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file, index_col=0)
    else:
        st.warning("Please upload a CSV to continue.")
        st.stop()
else:
    n_symbols = st.sidebar.slider("Number of symbols", 3, 12, 6)
    data = generate_sample_data(n_symbols)

st.subheader("Symbolic Data Matrix")
st.dataframe(data.style.background_gradient(cmap="viridis"), use_container_width=True)

# --- Resonance Matrix Computation ---
matrix = compute_resonance_matrix(data)

st.subheader("Resonance Matrix")
st.dataframe(matrix.style.background_gradient(cmap="plasma"), use_container_width=True)

# --- Resonant Clusters ---
clusters = find_resonant_clusters(matrix)
st.subheader("Resonant Clusters")
for i, cluster in enumerate(clusters, start=1):
    st.write(f"**Cluster {i}:** {', '.join(cluster)}")

st.subheader("3D Resonance Field Visualization")

try:
    fig = render_3d_resonance_field(matrix, clusters)
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Visualization failed: {e}")

from modules.analytics import compute_symbol_energy
from src.vector_plot import render_3d_energy_field

# ---Compute and visualize energy-frequency field ---
energy_df = compute_symbol_energy(data)
st.subheader("⚡ Symbolic Energy–Frequency Field")
fig_energy = render_3d_energy_field(data, energy_df)
st.plotly_chart(fig_energy, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>"
    "IVC Symbolic Visualizer © 2025 – Built for exploratory mapping and symbolic coherence research."
    "</p>",
    unsafe_allow_html=True
)
