import streamlit as st
import numpy as np
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

# --- Session State for Animation ---
if "is_playing" not in st.session_state:
    st.session_state.is_playing = {"resonance": False, "flow": False, "network": False}
if "step" not in st.session_state:
    st.session_state.step = 0

# --- Control Functions ---
def toggle_play(chart_name):
    st.session_state.is_playing[chart_name] = not st.session_state.is_playing[chart_name]

def increment_step():
    st.session_state.step += 1
    return st.session_state.step


# --- Layout ---
tab1, tab2, tab3 = st.tabs(["3D Resonance Field", "Energy Flow", "Symbolic Network"])

# --- Resonance Field ---
with tab1:
    st.subheader("Resonance Field Animation")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
            "▶️ Play" if not st.session_state.is_playing["resonance"] else "⏸ Pause",
            key="resonance_btn",
        ):
            toggle_play("resonance")

    with col2:
        current_step = increment_step() if st.session_state.is_playing["resonance"] else st.session_state.step
        evolved_matrix = evolve_matrix_step(matrix, current_step)
        fig_resonance = render_3d_resonance_field(evolved_matrix, clusters)
        st.plotly_chart(fig_resonance, use_container_width=True, key=f"resonance_chart_{current_step}")

# --- Energy Flow Field ---
with tab2:
    st.subheader("Energy Flow Field Animation")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
            "▶️ Play" if not st.session_state.is_playing["flow"] else "⏸ Pause",
            key="flow_btn",
        ):
            toggle_play("flow")

    with col2:
        current_step = increment_step() if st.session_state.is_playing["flow"] else st.session_state.step
        evolved_matrix = evolve_matrix_step(matrix, current_step)
        flow_vectors = compute_energy_flow(evolved_matrix)
        fig_flow = render_energy_flow_field(evolved_matrix, flow_vectors)
        st.plotly_chart(fig_flow, use_container_width=True, key=f"flow_chart_{current_step}")

# --- Symbolic Network ---
with tab3:
    st.subheader("Symbolic Network Dynamics")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(
            "▶️ Play" if not st.session_state.is_playing["network"] else "⏸ Pause",
            key="network_btn",
        ):
            toggle_play("network")

    with col2:
        current_step = increment_step() if st.session_state.is_playing["network"] else st.session_state.step
        evolved_matrix = evolve_matrix_step(matrix, current_step)
        fig_network = render_symbolic_network(evolved_matrix)
        st.plotly_chart(fig_network, use_container_width=True, key=f"network_chart_{current_step}")

st.info("Use the Play/Pause buttons above to animate each chart independently.")
