import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- Ensure /src directory is accessible for imports ---
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODULES_DIR = BASE_DIR / "modules"
for p in [SRC_DIR, MODULES_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.append(str(p))

# --- Imports from our modules ---
try:
    from modules.analytics import (
        generate_sample_data,
        compute_resonance_matrix,
        find_resonant_clusters
    )
except ImportError as e:
    st.error(f"ImportError: {e}")
    st.stop()

# --- Optionally import SRC helpers if present ---
try:
    from src.resonance_tools import resonance_matrix as src_resonance_matrix
    USE_SRC_RESONANCE = True
except Exception:
    USE_SRC_RESONANCE = False

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="Symbolic Resonance Visualizer",
    page_icon="🔆",
    layout="wide"
)

st.title("🔆 Symbolic Resonance Visualizer")
st.caption("Dynamic symbolic field mapping and resonance clustering")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
num_symbols = st.sidebar.slider("Number of symbols", 3, 12, 6)
threshold = st.sidebar.slider("Resonance threshold", 0.0, 1.0, 0.8)

# --- Data Generation ---
data = generate_sample_data(num_symbols)

# --- Resonance Computation (choosing best available source) ---
if USE_SRC_RESONANCE:
    st.sidebar.success("Using resonance_matrix from /src tools")
    matrix = src_resonance_matrix(data)
else:
    st.sidebar.info("Using compute_resonance_matrix from modules.analytics")
    matrix = compute_resonance_matrix(data)

clusters = find_resonant_clusters(matrix, threshold=threshold)

# --- Display Data ---
st.subheader("Input Data")
st.dataframe(data)

st.subheader("Resonance Matrix")
st.dataframe(matrix.style.background_gradient(cmap="viridis"))

st.subheader("Resonant Clusters")
for c in clusters:
    st.write(", ".join(sorted(list(c))))

# --- 3D Visualization Placeholder ---
st.subheader("3D Resonance Field (coming next)")
st.info("The 3D resonance field will render symbol relationships in spatial form.")
# When visuals are ready:
# from modules.visuals import render_resonance_field
# render_resonance_field(matrix)

st.markdown("---")
st.caption("IVC Symbolic Visualizer · Streamlit Experimental Build")
