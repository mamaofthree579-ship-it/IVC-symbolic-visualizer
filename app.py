import streamlit as st
import numpy as np
import time
from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
    compute_symbol_energy,
    compute_energy_flow,
    evolve_matrix_step,
)
from src.vector_plot import (
    render_3d_resonance_field,
    render_energy_flow_field,
    render_symbolic_network,
)

st.set_page_config(page_title="IVC Symbolic Visualizer", layout="wide")
st.title("🌐 IVC Symbolic Energy Visualizer")

# --- Initialize Data ---
data = generate_sample_data(8)
matrix = compute_resonance_matrix(data)
clusters = find_resonant_clusters(matrix)
flow_vectors = compute_energy_flow(data)

# --- Session State ---
if "is_playing" not in st.session_state:
    st.session_state.is_playing = {"resonance": False, "flow": False, "network": False}
if "step" not in st.session_state:
    st.session_state.step = 0

def toggle_play(chart_name):
    st.session_state.is_playing[chart_name] = not st.session_state.is_playing[chart_name]

def increment_step():
    st.session_state.step += 1
    return st.session_state.step

def auto_loop(chart_name, limit=30, delay=0.3):
    """Automatically advance frames until paused or limit reached."""
    for _ in range(limit):
        if not st.session_state.is_playing.get(chart_name):
            break
        st.session_state.step += 1
        time.sleep(delay)
        st.experimental_rerun()

# --- Layout ---
tab1, tab2, tab3 = st.tabs(["3D Resonance Field", "Energy Flow", "Symbolic Network"])

# --- Resonance Field ---
with tab1:
    st.subheader("Resonance Field Animation")
    if st.button("▶️ Play" if not st.session_state.is_playing["resonance"] else "⏸ Pause", key="res_btn"):
        toggle_play("resonance")
        if st.session_state.is_playing["resonance"]:
            auto_loop("resonance")

    step = st.session_state.step
    evolved = evolve_matrix_step(matrix, step)
    fig_res = render_3d_resonance_field(evolved, clusters)
    st.plotly_chart(fig_res, use_container_width=True, key=f"resonance_{step}")

# --- Energy Flow Field ---
with tab2:
    st.subheader("Energy Flow Field Animation")
    if st.button("▶️ Play" if not st.session_state.is_playing["flow"] else "⏸ Pause", key="flow_btn"):
        toggle_play("flow")
        if st.session_state.is_playing["flow"]:
            auto_loop("flow")

    step = st.session_state.step
    evolved = evolve_matrix_step(matrix, step)
    flow = compute_energy_flow(evolved)
    fig_flow = render_energy_flow_field(evolved, flow)
    st.plotly_chart(fig_flow, use_container_width=True, key=f"flow_{step}")

# --- Symbolic Network ---
with tab3:
    st.subheader("Symbolic Network Dynamics")
    if st.button("▶️ Play" if not st.session_state.is_playing["network"] else "⏸ Pause", key="net_btn"):
        toggle_play("network")
        if st.session_state.is_playing["network"]:
            auto_loop("network")

    step = st.session_state.step
    evolved = evolve_matrix_step(matrix, step)
    fig_net = render_symbolic_network(evolved)
    st.plotly_chart(fig_net, use_container_width=True, key=f"network_{step}")
