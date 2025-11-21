import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Indus Resonance Lab", layout="centered")
st.title("Indus Resonance Lab — Built-In Symbols")

# ----------- File Paths -----------
BUILT_INS = {
    "Jar": "Phone/jar.jpg",  # You uploaded this file
}

# If we want to slice the 3 symbols from the sheet:
def load_symbol_from_sheet(path, index):
    img = Image.open(path).convert("L")
    w, h = img.size
    left = int(index * w / 3)
    right = int((index + 1) * w / 3)
    crop = img.crop((left, 0, right, h))
    crop = crop.resize((512,512))
    return crop

# Auto load 3 symbols from uploaded file
images = {
    "Jar": load_symbol_from_file ("Phone/jar.jpg"),
    "Fish": load_symbol_from_file("Phone/fish.jpg"),
    "Double Fish": load_symbol_from_file("Phone/double_fish.jpg")
}

# ----------- Symbol Selection -----------
choice = st.selectbox("Choose symbol", list(symbols.keys()))
image = symbols[choice]
st.image(image, caption=f"{choice} Symbol", use_column_width=True)

arr = np.array(image).astype(np.float32)

# ---------------- FFT ----------------
st.subheader("FFT Resonance Map")
fft = np.fft.fft2(arr)
fft_shift = np.fft.fftshift(fft)
mag = np.log(np.abs(fft_shift) + 1)
st.image(mag/mag.max(), use_column_width=True)

# Frequency slider
freq = st.slider("Activation Frequency (Hz)", 1, 100, 30)

# Wave resonance model
y, x = np.indices(arr.shape)
wave = np.sin(2*np.pi*freq * x / arr.shape[1])
resonance = (arr/255.0) * (wave + 1)

st.subheader(f"Resonance Output @ {freq} Hz")
st.image(resonance, use_column_width=True)
