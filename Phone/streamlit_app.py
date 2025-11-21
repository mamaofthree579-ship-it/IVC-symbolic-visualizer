import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Indus Resonance System", layout="centered")

st.title("Indus Symbol Resonance • Full System")
st.markdown("""
This system performs:
- FFT Resonance Extraction  
- Activation Frequency Response  
- Holographic Geometry Classification  
- Multi-symbol Harmonic Composition  
- Sequence Engine (A → B → C → Combined signature)  
""")

# ---------------------------------------------------------
# 1 — BUILT-IN SYMBOLS
# ---------------------------------------------------------
# Update these filenames to match YOUR uploaded JPGs
BUILT_INS = {
    "Jar": "A_digital_vector_image_displays_three_black_Indus_.jpg",
    "Fish": "indus_fish.jpg",
    "Double Fish": "indus_double_fish.jpg"
}

def load_image(path):
    return Image.open(path).convert("L")

def load_or_warn(path):
    if os.path.exists(path):
        return load_image(path)
    else:
        st.error(f"Missing file: {path}")
        return None

st.header("Built-In Symbols")

symbol_names = list(BUILT_INS.keys())
choice = st.selectbox("Choose a symbol", symbol_names)

img_path = BUILT_INS[choice]
img = load_or_warn(img_path)

if img is not None:
    st.image(img, caption=choice, use_column_width=True)
    arr = np.array(img).astype(np.float32)

    # ---------------------------------------------------------
    # 2 — FFT Resonance Map
    # ---------------------------------------------------------
    st.subheader("FFT Resonance Map")

    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1)

    st.image(mag / mag.max(), caption="FFT Amplitude", use_column_width=True)

    # ---------------------------------------------------------
    # 3 — Frequency Activation Simulation
    # ---------------------------------------------------------
    st.subheader("Frequency Activation")

    freq = st.slider("Activation frequency (Hz)", 1, 100, 33)

    y, x = np.indices(arr.shape)
    wave = np.sin(2 * np.pi * freq * x / arr.shape[1])
    resonance = (arr / 255.0) * (wave + 1)

    st.image(resonance, caption=f"Resonance @ {freq} Hz", use_column_width=True)

    # ---------------------------------------------------------
    # 4 — Holographic Geometry Analysis
    # ---------------------------------------------------------
    st.header("Holographic Geometry Classification")

    norm = mag / mag.max()

    # Vertical symmetry
    v_sym = np.sum(np.abs(norm - np.flip(norm, axis=1)))
    v_score = 1 - (v_sym / norm.size)

    # Horizontal symmetry
    h_sym = np.sum(np.abs(norm - np.flip(norm, axis=0)))
    h_score = 1 - (h_sym / norm.size)

    # Radial lobes
    center = (norm.shape[0]//2, norm.shape[1]//2)
    angles = np.arctan2(
        *np.indices(norm.shape)[::-1] - np.array(center)[:,None,None]
    )

    radial_bins = np.linspace(-np.pi, np.pi, 32)
    hist, _ = np.histogram(angles, bins=radial_bins, weights=norm)
    lobe_count = (hist > 0.3 * hist.max()).sum()

    # Rotational harmonics
    fft1d = np.abs(np.fft.fft(hist))
    rot_harmonics = (fft1d > 0.2 * fft1d.max()).sum()

    # Code
    geo_code = f"G{lobe_count}-V{v_score:.2f}-H{h_score:.2f}-R{rot_harmonics}"
    st.subheader("Geometry Code")
    st.code(geo_code)

# ---------------------------------------------------------
# 5 — MULTI-SYMBOL HARMONIC COMPOSER
# ---------------------------------------------------------
st.header("Multi-Symbol Harmonic Composer")

multi_files = st.file_uploader(
    "Upload 2–3 JPGs for harmonic blending:",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if multi_files:
    imgs = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in multi_files]
    shapes = set([im.shape for im in imgs])

    if len(shapes) > 1:
        st.error("All images must have same size.")
    else:
        stacked = np.stack(imgs, axis=0)

        st.subheader("Adjust Harmonic Weights")
        weights = []
        for i, f in enumerate(multi_files):
            w = st.slider(f"Harmonic weight for {f.name}", 0.0, 3.0, 1.0, 0.1)
            weights.append(w)

        weights = np.array(weights).reshape(len(weights),1,1)
        combined = np.sum(stacked * weights, axis=0)

        st.subheader("Combined Harmonic Signature")
        st.image(combined/combined.max(), use_column_width=True)

# ---------------------------------------------------------
# 6 — SEQUENCE ENGINE (A → B → C → Composite)
# ---------------------------------------------------------
st.header("Sequence Engine")

seq_files = st.file_uploader(
    "Upload up to 3 symbols in sequence order:",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if seq_files:
    imgs = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in seq_files]

    if len(set([im.shape for im in imgs])) > 1:
        st.error("All images must have same size.")
    else:
        st.write("Sequence:", " → ".join([f.name for f in seq_files]))

        # Combine sequentially
        out = imgs[0] / 255.0
        for i, im in enumerate(imgs[1:], start=2):
            out = np.sin(out + im/255.0)

        st.subheader("Sequence Output")
        st.image(out, use_column_width=True)
