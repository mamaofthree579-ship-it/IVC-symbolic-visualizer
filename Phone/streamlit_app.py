import streamlit as st
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Indus Symbol Resonance Lab", layout="centered")

st.title("Indus Symbol Resonance Lab")
st.markdown("Upload a symbol PNG to analyze its frequency response.")

# --- Upload zone ---
uploaded_file = st.file_uploader("Upload a symbol PNG", type=["png","jpg","jpeg"])

if uploaded_file:
    # load image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Symbol", use_column_width=True)

    # convert to numpy
    arr = np.array(image).astype(np.float32)

    # --- FFT ---
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1)

    st.subheader("FFT Resonance Map")
    st.image(mag / mag.max(), caption="Frequency-domain amplitude", use_column_width=True)

    # --- Frequency slider ---
    freq = st.slider("Activation Frequency (Hz)", 1, 100, 30)

    # simple synthetic model: resonance = symbol * sine wave
    y, x = np.indices(arr.shape)
    wave = np.sin(2*np.pi*freq * x/arr.shape[1])
    resonance = (arr / 255.0) * (wave + 1.0)

    st.subheader("Resonance Output")
    st.image(resonance, caption=f"Response at {freq} Hz", use_column_width=True)

# ------------- Multi-symbol math model -------------
st.header("Multi-Symbol Harmonic Composer")

uploaded_files = st.file_uploader(
    "Upload 2–3 PNGs to combine",
    type=["png","jpg","jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    images = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in uploaded_files]
    shapes = list(set([img.shape for img in images]))

    if len(shapes) > 1:
        st.error("All images must have the same size.")
    else:
        stacked = np.stack(images, axis=0)

        # harmonic weights
        st.subheader("Symbol Harmonic Weights")
        weights = []
        for i, f in enumerate(uploaded_files):
            w = st.slider(f"Weight for {f.name}", 0.0, 2.0, 1.0, 0.1)
            weights.append(w)

        weights = np.array(weights).reshape(-1, 1, 1)
        combined = np.sum(stacked * weights, axis=0)

        st.subheader("Combined Harmonic Signature")
        st.image(combined / combined.max(), use_column_width=True)
