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
    render_symbolic_network
)

# === Streamlit App ===
st.set_page_config(page_title="IVC Symbolic Visualizer", layout="wide")

st.title("🌐 IVC Symbolic Visualizer")
st.caption("Exploring resonance, energy flow, and symbolic network relationships")

# --- Data Generation ---
st.sidebar.header("Data Controls")
num_symbols = st.sidebar.slider("Number of Symbols", 3, 15, 7)
threshold = st.sidebar.slider("Resonance Threshold", 0.3, 0.95, 0.6, 0.05)

if "data" not in st.session_state:
    st.session_state.data = generate_sample_data(num_symbols)

if st.sidebar.button("🔄 Regenerate Data"):
    st.session_state.data = generate_sample_data(num_symbols)

data = st.session_state.data
st.subheader("📊 Symbolic Data")
st.dataframe(data.style.background_gradient(cmap="viridis"), use_container_width=True)

# --- Resonance Matrix ---
st.subheader("🔮 Resonance Matrix")
matrix = compute_resonance_matrix(data)
st.dataframe(matrix.style.background_gradient(cmap="plasma"), use_container_width=True)

clusters = find_resonant_clusters(matrix, threshold)

# --- 3D Resonance Visualization ---
st.subheader("🌌 3D Resonance Field")
try:
    fig_resonance = render_3d_resonance_field(matrix, clusters)
    st.plotly_chart(fig_resonance, use_container_width=True)
except Exception as e:
    st.error(f"Resonance field visualization failed: {e}")

# --- Energy Flow Visualization ---
st.subheader("⚡ Energy Flow Field")
try:
    flow_vectors = compute_energy_flow(data)
    fig_energy = render_energy_flow_field(data, flow_vectors)
    st.plotly_chart(fig_energy, use_container_width=True)
except Exception as e:
    st.error(f"Energy flow visualization failed: {e}")

# --- Symbolic Network Visualization ---
st.subheader("🌐 Symbolic Network Connectivity")
try:
    fig_network = render_symbolic_network(matrix, threshold=threshold)
    st.plotly_chart(fig_network, use_container_width=True)
except Exception as e:
    st.error(f"Symbolic network visualization failed: {e}")

# --- Symbol Energy Mapping ---
st.subheader("💠 Symbol Energy Mapping")
try:
    energy = compute_symbol_energy(data)
    energy_df = pd.DataFrame(energy, columns=["Energy"])
    st.bar_chart(energy_df)
except Exception as e:
    st.error(f"Energy mapping failed: {e}")

st.markdown("---")
st.markdown("✨ *IVC Symbolic Visualizer – Dynamic Energy & Resonance System Prototype* ✨")
