import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import time
from pathlib import Path
import sys

# --- Streamlit setup
st.set_page_config(page_title="Symbolic Resonance Visualizer", layout="wide")
st.title("🌐 Symbolic Resonance Visualizer")

# --- Diagnostic heartbeat
st.markdown("### 🔄 Diagnostic Check")
st.write("App started successfully ✅")

# --- Path setup
BASE_DIR = Path(__file__).resolve().parent
MODULES_DIR = BASE_DIR / "modules"
SRC_DIR = BASE_DIR / "src"

for p in [MODULES_DIR, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

try:
    from modules.analytics import (
        generate_sample_data,
        compute_resonance_matrix,
        find_resonant_clusters,
    )
    st.success("✅ Module import successful.")
except Exception as e:
    st.error(f"Module import failed: {e}")
    st.stop()

# --- Sidebar controls
st.sidebar.header("Configuration")
num_symbols = st.sidebar.slider("Number of symbols", 3, 15, 6)
threshold = st.sidebar.slider("Resonance threshold", 0.0, 1.0, 0.8)

# --- Generate data
data = generate_sample_data(num_symbols)
matrix = compute_resonance_matrix(data)
clusters = find_resonant_clusters(matrix, threshold)

st.subheader("📊 Resonance Matrix")
st.dataframe(matrix)

# --- Visualization step
st.markdown("### 🖼 Rendering heatmap...")

try:
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
        )
    )
    fig.update_layout(title="Resonance Heatmap", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ Heatmap rendered successfully.")
except Exception as e:
    st.error(f"Heatmap error: {e}")

# --- Show clusters
st.markdown("### 🔹 Identified Clusters")
for c in clusters:
    st.write(", ".join(c))

st.markdown("---")
st.caption(f"Render complete — timestamp: {time.strftime('%X')}")
