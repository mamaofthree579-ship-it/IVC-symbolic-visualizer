import streamlit as st
import pandas as pd
import numpy as np

# Import analysis & visualization modules
from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
    compute_symbol_energy,
    compute_energy_flow,
)
from src.vector_plot import (
    render_3d_resonance_field,
    render_energy_flow_field,
    render_frequency_spectrum,
)

# ------------------------------------------------------------------------------
# Streamlit App Configuration
# ------------------------------------------------------------------------------
st.set_page_config(page_title="IVC Symbolic Energy Visualizer", layout="wide")

st.title("🌀 IVC Symbolic Energy Visualizer")
st.markdown("""
This tool helps researchers explore how **ancient symbolic systems** may encode
energetic and relational data.  
Each visualization is interactive and dynamically generated.
""")

# ------------------------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------------------------
st.sidebar.header("Controls")
num_symbols = st.sidebar.slider("Number of symbols", 3, 20, 8)
energy_threshold = st.sidebar.slider("Resonance threshold", 0.5, 0.95, 0.8)
show_energy_flow = st.sidebar.checkbox("Show Energy Flow Field", True)
show_frequency_spectrum = st.sidebar.checkbox("Show Frequency Spectrum", True)

# ------------------------------------------------------------------------------
# Data Generation and Core Computations
# ------------------------------------------------------------------------------
st.subheader("1️⃣ Generating Symbolic Data")
data = generate_sample_data(num_symbols)
st.dataframe(data, use_container_width=True)

st.subheader("2️⃣ Computing Resonance Matrix")
resonance_matrix = compute_resonance_matrix(data)
st.dataframe(resonance_matrix, use_container_width=True)

# Compute symbolic clusters
clusters = find_resonant_clusters(resonance_matrix, threshold=energy_threshold)

# ------------------------------------------------------------------------------
# Visualization Tabs
# ------------------------------------------------------------------------------
st.subheader("3️⃣ Visualizations")
tabs = st.tabs([
    "3D Resonance Field",
    "Energy Flow",
    "Frequency Spectrum"
])

# ------------------------------------------------------------------------------
# Tab 1 – 3D Resonance Field
# ------------------------------------------------------------------------------
with tabs[0]:
    st.markdown("### 🌐 Resonant Symbolic Clusters")
    try:
        fig_resonance = render_3d_resonance_field(resonance_matrix, clusters)
        st.plotly_chart(fig_resonance, use_container_width=True)
    except Exception as e:
        st.error(f"Resonance visualization failed: {e}")

# ------------------------------------------------------------------------------
# Tab 2 – Energy Flow Visualization
# ------------------------------------------------------------------------------
if show_energy_flow:
    with tabs[1]:
        st.markdown("### ⚡ Symbolic Energy Flow Field")
        try:
            flow_vectors = compute_energy_flow(data)
            fig_flow = render_energy_flow_field(data, flow_vectors)
            st.plotly_chart(fig_flow, use_container_width=True)
        except Exception as e:
            st.error(f"Energy flow visualization failed: {e}")

# ------------------------------------------------------------------------------
# Tab 3 – Frequency Spectrum
# ------------------------------------------------------------------------------
if show_frequency_spectrum:
    with tabs[2]:
        st.markdown("### 🔊 Symbolic Frequency Spectrum")
        try:
            symbol_energy = compute_symbol_energy(data)
            fig_freq = render_frequency_spectrum(symbol_energy)
            st.plotly_chart(fig_freq, use_container_width=True)
        except Exception as e:
            st.error(f"Frequency spectrum visualization failed: {e}")

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("""
---
💠 *Developed to support linguistic archaeology and energy pattern research.*  
Use responsibly with respect for all living systems.
""")
