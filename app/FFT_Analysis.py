import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from utils.fft_tools import spatial_fft
from utils.symbol_loader import load_symbol

st.title("📡 FFT & Resonance Analysis")

uploaded = st.file_uploader("Upload symbol heightmap (.npy)", type=["npy"])

height = load_symbol(uploaded)

fft_img = spatial_fft(height)

st.subheader("Symbol FFT")
fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(fft_img, cmap="inferno")
ax.axis("off")
st.pyplot(fig)
