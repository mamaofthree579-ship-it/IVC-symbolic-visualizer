# app.py
import streamlit as st
import time
import numpy as np
import pandas as pd

from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
    compute_energy_flow,
    compute_symbol_energy,
    evolve_matrix_step
)

from src.vector_plot import (
    render_3d_resonance_field,
    render_energy_flow_field,
    render_symbolic_network
)

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="IVC Living Resonance Visualizer", layout="wide")
st.title("🔄 IVC Living Resonance Visualizer — Auto-cycle Demo")

st.markdown(
    "This demo auto-animates the Resonance Matrix, Energy Flow Field, and Symbolic Network "
    "for a small number of cycles and then stops. Use the sidebar to regenerate or re-run."
)

# Sidebar controls
st.sidebar.header("Animation Controls")
n_symbols = st.sidebar.slider("Number of symbols", 4, 14, 8)
cycles = st.sidebar.number_input("Cycle count", min_value=1, max_value=60, value=10, step=1)
interval_seconds = st.sidebar.slider("Update interval (seconds)", 1.0, 5.0, 3.0, step=0.5)
amplitude = st.sidebar.slider("Energy fluctuation amplitude", 0.01, 0.5, 0.12, step=0.01)

if st.sidebar.button("Regenerate base data"):
    st.session_state.pop("base_data", None)
    st.session_state.pop("base_matrix", None)

# Ensure base data/matrix persisted between reruns
if "base_data" not in st.session_state:
    st.session_state.base_data = generate_sample_data(n_symbols)
if "base_matrix" not in st.session_state:
    st.session_state.base_matrix = compute_resonance_matrix(st.session_state.base_data)

base_data = st.session_state.base_data
base_matrix = st.session_state.base_matrix

# Layout: three columns for the three visualizations
col_res, col_flow, col_net = st.columns(3)

with col_res:
    st.subheader("🔮 Resonance Matrix (animated)")

with col_flow:
    st.subheader("⚡ Energy Flow Field (animated)")

with col_net:
    st.subheader("🌐 Symbolic Network (animated)")

# Placeholders so we can update in-place
placeholder_res = col_res.empty()
placeholder_flow = col_flow.empty()
placeholder_net = col_net.empty()
status_line = st.empty()

# Precompute static items
static_data = base_data.copy()

# Animation loop: run N cycles then stop
cycles = int(cycles)
interval = float(interval_seconds)

try:
    for t in range(cycles):
        # Produce a time-varying matrix from the base (small oscillatory perturbation)
        matrix_t = evolve_matrix_step(base_matrix, t, amplitude)

        # Recompute clusters based on the current matrix
        clusters_t = find_resonant_clusters(matrix_t, threshold=0.6)

        # For the energy/flow visuals, produce flow vectors using analytics
        # We'll scale input data by the instantaneous energy factor to reflect time-evolution
        energy_signal = compute_symbol_energy(pd.DataFrame(matrix_t.values, index=matrix_t.index, columns=matrix_t.columns))
        # Make a time-modulated data snapshot
        mod_factor = 1.0 + 0.5 * np.sin(2.0 * np.pi * (t / max(1, cycles)))
        snapshot_df = static_data * (1.0 + mod_factor * (energy_signal / (np.nanmax(energy_signal) + 1e-9))[:, None])

        flow_vectors_t = compute_energy_flow(snapshot_df)

        # Render each visual and place into its placeholder
        fig_res = render_3d_resonance_field(matrix_t, clusters_t)
        placeholder_res.plotly_chart(fig_res, use_container_width=True)

        fig_flow = render_energy_flow_field(snapshot_df, flow_vectors_t)
        placeholder_flow.plotly_chart(fig_flow, use_container_width=True)

        fig_net = render_symbolic_network(matrix_t, threshold=0.55)
        placeholder_net.plotly_chart(fig_net, use_container_width=True)

        status_line.info(f"Animation cycle {t+1}/{cycles} — updating every {interval:.1f}s")
        time.sleep(interval)

    status_line.success(f"Animation completed ({cycles} cycles). Use sidebar to re-run or regenerate.")
except Exception as e:
    status_line.error(f"Animation aborted due to error: {e}")
    raise
