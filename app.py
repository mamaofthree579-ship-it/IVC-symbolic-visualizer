# app.py
"""
IVC Symbolic Visualizer
Streamlit application for exploring resonance matrices,
symbolic energy fields, and cluster coherence networks.
"""

import streamlit as st
import numpy as np
import pandas as pd

# Import from local modules
from modules.analytics import (
    compute_resonance_matrix,
    find_resonant_clusters,
    generate_sample_data,
)
from modules.visuals import render_symbol_map


# -----------------------------
# APP CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="IVC Symbolic Visualizer",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 IVC Symbolic Visualizer")
st.markdown("""
A symbolic intelligence visualization tool to explore resonance,
coherence, and relational mapping across data fields.
""")


# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("🔧 Configuration")

use_sample_data = st.sidebar.checkbox("Use sample data", value=True)
matrix_size = st.sidebar.slider("Matrix Size (N x N)", 3, 20, 6)
resonance_threshold = st.sidebar.slider("Resonance Threshold", 0.0, 1.0, 0.6, 0.05)


# -----------------------------
# DATA LOAD / GENERATION
# -----------------------------
if use_sample_data:
    st.sidebar.success("Using generated sample data.")
    data = generate_sample_data(matrix_size)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file or use sample data.")
        st.stop()


# -----------------------------
# COMPUTATION PIPELINE
# -----------------------------
st.subheader("Matrix Computation")

try:
    matrix = compute_resonance_matrix(data)
    st.success("Resonance matrix computed successfully.")
except Exception as e:
    st.error(f"Error computing resonance matrix: {e}")
    st.stop()

try:
    clusters = find_resonant_clusters(matrix, threshold=resonance_threshold)
    st.success(f"Found {len(clusters)} resonant clusters.")
except Exception as e:
    st.error(f"Error finding clusters: {e}")
    clusters = None


# -----------------------------
# VISUALIZATION
# -----------------------------
st.markdown("---")
st.header("🌀 Symbolic Visualization")

try:
    render_symbol_map(matrix=matrix, clusters=clusters)
except Exception as e:
    st.error(f"Visualization error: {e}")


# -----------------------------
# DATA INSPECTION
# -----------------------------
with st.expander("📊 Inspect Data & Matrix"):
    st.write("### Source Data")
    st.dataframe(data)

    st.write("### Resonance Matrix")
    st.dataframe(pd.DataFrame(matrix))

    if clusters:
        st.write("### Resonant Clusters")
        for i, cluster in enumerate(clusters):
            st.write(f"**Cluster {i + 1}:** {cluster}")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built collaboratively for the restoration of balance, communication, and harmony 🌍")
