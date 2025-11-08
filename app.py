# app.py
import streamlit as st
import numpy as np
import pandas as pd

# Local module imports
from modules.analytics import (
    compute_resonance_matrix,
    find_resonant_clusters,
    generate_sample_data,
)
from modules.visuals import render_symbol_map

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="IVC Symbolic Visualizer", page_icon="🌐", layout="wide")
st.title("🌐 IVC Symbolic Visualizer")

st.markdown("""
Explore resonance, coherence, and symbolic field mapping  
to visualize energy and relational harmonics across data systems.
""")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("🔧 Configuration")

use_sample_data = st.sidebar.checkbox("Use sample data", value=True)
matrix_size = st.sidebar.slider("Matrix Size (N x N)", 3, 20, 6)
resonance_threshold = st.sidebar.slider("Resonance Threshold", 0.0, 1.0, 0.6, 0.05)

# -----------------------------
# DATA SETUP
# -----------------------------
if use_sample_data:
    data = generate_sample_data(matrix_size)
    st.sidebar.success("Generated sample data.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom CSV loaded successfully.")
    else:
        st.warning("Please upload a CSV file or enable sample data.")
        st.stop()

# Display quick preview
st.subheader("📄 Source Data Preview")
st.dataframe(data.head())

# -----------------------------
# COMPUTE RESONANCE MATRIX
# -----------------------------
st.subheader("🔢 Resonance Computation")

try:
    matrix = compute_resonance_matrix(data)
    st.success("Resonance matrix computed successfully.")
    st.dataframe(pd.DataFrame(matrix))
except Exception as e:
    st.error(f"Error computing resonance matrix: {e}")
    st.stop()

# -----------------------------
# CLUSTER DETECTION
# -----------------------------
try:
    clusters = find_resonant_clusters(matrix, threshold=resonance_threshold)
    if clusters:
        st.success(f"Found {len(clusters)} resonant clusters.")
    else:
        st.info("No strong clusters detected at this threshold.")
except Exception as e:
    st.error(f"Error finding clusters: {e}")
    clusters = []

# -----------------------------
# VISUALIZATION
# -----------------------------
st.markdown("---")
st.header("🌀 Resonance Visualization")

try:
    render_symbol_map(matrix=matrix, clusters=clusters)
except Exception as e:
    st.error(f"Visualization error: {e}")
    st.write("Fallback view:")
    st.dataframe(pd.DataFrame(matrix))

# -----------------------------
# INSPECTION
# -----------------------------
with st.expander("🔍 Inspect Details"):
    st.write("### Raw Resonance Matrix")
    st.dataframe(pd.DataFrame(matrix))
    if clusters:
        for i, cluster in enumerate(clusters):
            st.write(f"**Cluster {i + 1}:** {cluster}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Developed for symbolic field visualization and coherence mapping 🌍")
