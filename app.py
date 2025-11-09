import streamlit as st
import numpy as np
import time
from modules.analytics import (
    generate_sample_data,
    compute_resonance_matrix,
    find_resonant_clusters,
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

# --- Session State ---
if "is_playing" not in st.session_state:
    st.session_state.is_playing = {"resonance": False, "flow": False, "network": False}
if "frame" not in st.session_state:
    st.session_state.frame = 0

# --- Animation helper ---
def toggle_play(chart):
    st.session_state.is_playing[chart] = not st.session_state.is_playing[chart]
    st.session_state.frame = 0


def animate_chart(chart_name, render_func, *args):
    """Animate chart locally for up to 30 frames."""
    placeholder = st.empty()
    for i in range(30):
        if not st.session_state.is_playing[chart_name]:
            break
        evolved_matrix = evolve_matrix_step(matrix, i)
        fig = render_func(evolved_matrix, *args)
        placeholder.plotly_chart(fig, use_container_width=True, key=f"{chart_name}_{i}")
        time.sleep(0.3)
    st.session_state.is_playing[chart_name] = False


# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["3D Resonance Field", "Energy Flow", "Symbolic Network"])

with tab1:
    st.subheader("3D Resonance Field")
    play_button = st.button(
        "▶️ Play" if not st.session_state.is_playing["resonance"] else "⏸ Pause",
        key="play_resonance",
    )
    if play_button:
        toggle_play("resonance")

    if st.session_state.is_playing["resonance"]:
        animate_chart("resonance", render_3d_resonance_field, clusters)
    else:
        st.plotly_chart(
            render_3d_resonance_field(matrix, clusters),
            use_container_width=True,
            key="resonance_static",
        )

with tab2:
    st.subheader("Energy Flow Field")
    play_button = st.button(
        "▶️ Play" if not st.session_state.is_playing["flow"] else "⏸ Pause",
        key="play_flow",
    )
    if play_button:
        toggle_play("flow")

    if st.session_state.is_playing["flow"]:
        animate_chart("flow", render_energy_flow_field, compute_energy_flow(matrix))
    else:
        st.plotly_chart(
            render_energy_flow_field(matrix, compute_energy_flow(matrix)),
            use_container_width=True,
            key="flow_static",
        )

with tab3:
    st.subheader("Symbolic Network")
    play_button = st.button(
        "▶️ Play" if not st.session_state.is_playing["network"] else "⏸ Pause",
        key="play_network",
    )
    if play_button:
        toggle_play("network")

    if st.session_state.is_playing["network"]:
        animate_chart("network", render_symbolic_network)
    else:
        st.plotly_chart(
            render_symbolic_network(matrix),
            use_container_width=True,
            key="network_static",
        )

st.caption("Each chart auto-plays up to 30 frames and then stops automatically.")
