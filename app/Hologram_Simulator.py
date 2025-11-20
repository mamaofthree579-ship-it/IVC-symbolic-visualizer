import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.hologram_tools import fresnel_propagation
from utils.symbol_loader import load_symbol

st.title("🔮 Holographic Interference Simulator")

uploaded = st.file_uploader("Upload symbol (.npy)", type=["npy"])
height = load_symbol(uploaded)

z = st.slider("Propagation Distance (m)", 0.05, 0.5, 0.15)

intensity = fresnel_propagation(height, z=z)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(intensity, cmap="magma")
ax.axis("off")
st.pyplot(fig)
