import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

# ================================================================
# ✅ Environment Setup
# ================================================================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODULES_DIR = BASE_DIR / "modules"

for path in [SRC_DIR, MODULES_DIR]:
    if path.exists() and str(path) not in sys.path:
        sys.path.append(str(path))

# ================================================================
# ✅ Module Imports
# ================================================================
try:
    from modules.analytics import (
        generate_sample_data,
        compute_resonance_matrix,
        find_resonant_clusters,
    )
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Optional imports from /src if present
try:
    from src.vector_data import load_vector_data
    from src.vector_plot import plot_vector_field
    USE_SRC = True
except Exception:
    USE_SRC = False

# ================================================================
# ✅ Streamlit Configuration
# ================================================================
st.set_page_config(
    page_title="Symbolic Resonance Visualizer",
    page_icon="🔆",
    layout="wide",
)

st.title("🔆 Symbolic Resonance Visualizer")
st.caption("Dynamic symbolic field mapping and resonance clustering")

# ================================================================
# ✅ Sidebar Controls
# ================================================================
st.sidebar.header("Settings")
num_symbols = st.sidebar.slider("Number of symbols", 3, 12, 6)
threshold = st.sidebar.slider("Resonance threshold", 0.0, 1.0, 0.8)
use_real_vectors = st.sidebar.checkbox("Use Vector Data (from src)", value=False)

# ================================================================
# ✅ Data Generation / Loading
# ================================================================
@st.cache_data
def load_symbol_data(n, use_vectors):
    if use_vectors and USE_SRC:
        try:
            return load_vector_data()
        except Exception as e:
            st.warning(f"Could not load vector data: {e}")
            return generate_sample_data(n)
    else:
        return generate_sample_data(n)

data = load_symbol_data(num_symbols, use_real_vectors)

# ================================================================
# ✅ Resonance Computation (Cached)
# ================================================================
@st.cache_data
def compute_resonance(df):
    start = time.time()
    matrix = compute_resonance_matrix(df)
    runtime = time.time() - start
    return matrix, runtime

matrix, runtime = compute_resonance(data)

# ================================================================
# ✅ Cluster Detection
# ================================================================
clusters = find_resonant_clusters(matrix, threshold=threshold)

# ================================================================
# ✅ Display
# ================================================================
st.sidebar.write(f"Computation time: {runtime:.3f}s")
st.sidebar.write(f"Matrix shape: {matrix.shape}")
st.sidebar.write(f"Detected clusters: {len(clusters)}")

st.subheader("Input Data")
st.dataframe(data)

st.subheader("Resonance Matrix")
st.dataframe(matrix.style.background_gradient(cmap="viridis"))

st.subheader("Resonant Clusters")
if clusters:
    for c in clusters:
        st.write("• " + ", ".join(sorted(list(c))))
else:
    st.info("No clusters found at current threshold.")

# ================================================================
# ✅ 3D / Vector Visualization Placeholder
# ================================================================
st.subheader("3D Resonance Field")
if USE_SRC and use_real_vectors:
    try:
        fig = plot_vector_field(data)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"3D plot unavailable: {e}")
else:
    st.info("3D visualization will activate when vector data is enabled.")

st.markdown("---")
st.caption("IVC Symbolic Visualizer · Experimental Streamlit Build")
