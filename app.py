import streamlit as st
import pandas as pd
import numpy as np

# --- IVC Modules ---
from src.data_loader import load_symbol_dataset, generate_synthetic_symbols
from src.energy_embedding import (
    normalize_features,
    compute_energy_vectors,
    compute_resonance_matrix,
)
from modules.energy_core import (
    evolve_energy_step,
    compute_energy_density,
    detect_energy_stabilization,
)
from src.ui_controls import play_pause_controls
from src.vector_plot import render_3d_resonance_field, render_energy_flow_field

# --- Streamlit Page Setup ---
st.set_page_config(page_title="IVC Symbolic Energy Visualizer", layout="wide")
st.title("🌐 IVC Symbolic Energy & Resonance Visualizer")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔹 Data & Preparation", "🔸 Energy Mapping", "🔻 Resonance Simulation"])

# ---------------------------------------------------------------------
# TAB 1: Load & Prepare Symbol Data
# ---------------------------------------------------------------------
with tab1:
    st.header("🔹 Symbol Dataset Loader")
    st.write("Upload or generate sample symbolic datasets for analysis.")

    data_option = st.radio("Choose Dataset Source:", ["Generate Sample", "Upload CSV"])

    if data_option == "Generate Sample":
        n = st.slider("Number of Symbols", 3, 20, 8)
        df = generate_synthetic_symbols(n)
    else:
        uploaded = st.file_uploader("Upload your CSV dataset", type=["csv", "tsv"])
        if uploaded:
            df = load_symbol_dataset(uploaded)
        else:
            st.warning("Please upload a dataset to continue.")
            st.stop()

    st.dataframe(df, use_container_width=True)

    st.success("✅ Dataset loaded successfully.")

# ---------------------------------------------------------------------
# TAB 2: Energy Mapping and Visualization
# ---------------------------------------------------------------------
with tab2:
    st.header("🔸 Symbolic Energy Mapping")
    st.write("Converts symbolic geometry into energetic representation fields.")

    normalized_df = normalize_features(df)
    energy_values = compute_energy_vectors(normalized_df)
    resonance_matrix = compute_resonance_matrix(normalized_df)

    df["energy"] = energy_values

    st.subheader("Energy Table")
    st.dataframe(df, use_container_width=True)

    st.subheader("Energy Density Distribution")
    st.bar_chart(df.set_index("symbol")["energy"])

    st.subheader("Resonance Field (3D Visualization)")
    fig = render_3d_resonance_field(resonance_matrix, None)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# TAB 3: Resonance Simulation (Dynamic)
# ---------------------------------------------------------------------
with tab3:
    st.header("🔻 Resonance Evolution Simulation")
    st.write("Simulate how symbolic energies evolve and stabilize through resonance coupling.")

    play, pause, reset = play_pause_controls("energy_sim")

    # Initialize or store energy matrix in session state
    if "energy_matrix" not in st.session_state or reset:
        st.session_state.energy_matrix = resonance_matrix.copy()
        st.session_state.history = [resonance_matrix.copy()]

    if play:
        new_matrix = evolve_energy_step(st.session_state.energy_matrix)
        st.session_state.energy_matrix = new_matrix
        st.session_state.history.append(new_matrix)

    # Compute energy density for display
    densities = compute_energy_density(st.session_state.energy_matrix)
    stabilized = detect_energy_stabilization(st.session_state.history)
from modules.analytics import find_resonant_clusters

clusters = find_resonant_clusters(resonance_matrix)
fig = render_3d_resonance_field(resonance_matrix, clusters)

    st.write(f"**Stabilization detected:** {'✅ Yes' if stabilized else '⏳ Not yet'}")

    st.subheader("Energy Density by Symbol")
    st.bar_chart(densities)

    st.subheader("Evolving Resonance Field (3D)")
    fig_field = render_energy_flow_field(df, st.session_state.energy_matrix)
    st.plotly_chart(fig_field, use_container_width=True)
