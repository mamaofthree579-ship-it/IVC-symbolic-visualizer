import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import time

st.set_page_config(page_title="Symbolic Resonance Visualizer", layout="wide")
st.title("🔆 Symbolic Resonance Visualizer (Debug Mode)")

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODULES_DIR = BASE_DIR / "modules"

for p in [SRC_DIR, MODULES_DIR]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

# --- basic progress feedback
st.write("✅ Imports and paths configured")

# --- safe imports
try:
    from modules.analytics import (
        generate_sample_data,
        compute_resonance_matrix,
        find_resonant_clusters,
    )
    st.write("✅ Analytics module loaded")
except Exception as e:
    st.error(f"❌ Failed to import analytics: {e}")
    st.stop()

# --- parameters
num_symbols = st.sidebar.slider("Number of symbols", 3, 10, 5)
threshold = st.sidebar.slider("Resonance threshold", 0.0, 1.0, 0.8)

st.write("✅ Sidebar loaded")

# --- generate data
try:
    st.write("⏳ Generating sample data...")
    df = generate_sample_data(num_symbols)
    st.write("✅ Data generated")
    st.dataframe(df)
except Exception as e:
    st.error(f"❌ Data generation failed: {e}")
    st.stop()

# --- compute resonance
try:
    st.write("⏳ Computing resonance matrix...")
    start = time.time()
    matrix = compute_resonance_matrix(df)
    st.write(f"✅ Resonance computed in {time.time()-start:.3f}s")
    st.dataframe(matrix)
except Exception as e:
    st.error(f"❌ Resonance computation failed: {e}")
    st.stop()

# --- clusters
try:
    st.write("⏳ Finding clusters...")
    clusters = find_resonant_clusters(matrix, threshold)
    st.write("✅ Clusters:", clusters)
except Exception as e:
    st.error(f"❌ Cluster detection failed: {e}")
    st.stop()

st.success("🎉 Debug complete — Streamlit rendered successfully!")
