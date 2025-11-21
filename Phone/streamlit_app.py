import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Indus Resonance System", layout="centered")

st.title("Indus Symbol Resonance • Full System")

# ======================================================
#  SAFETY NORMALIZER — FIXES ALL RUNTIMEERRORS
# ======================================================
def safe_image(arr):
    arr = np.nan_to_num(arr)
    arr = arr - arr.min()
    maxv = arr.max()
    if maxv > 0:
        arr = arr / maxv
    return arr

# ======================================================
# 1 — BUILT-IN SYMBOLS  
# UPDATE THESE FILENAMES TO MATCH YOUR JPG UPLOADS
# ======================================================
BUILT_INS = {
    # Replace these strings with the *exact* filenames you uploaded to Streamlit
    "Jar": "Phone/jar.jpg",
    "Fish": "Phone/fish.jpg",
    "Double Fish": "Phone/double_fish.jpg"
}

def load_image(path):
    return Image.open(path).convert("L")

def load_or_warn(path):
    if os.path.exists(path):
        return load_image(path)
    else:
        st.error(f"Missing file: {path}")
        return None

# ======================================================
# SYMBOL SELECTION
# ======================================================
st.header("Built-In Symbols")

symbol_names = list(BUILT_INS.keys())
choice = st.selectbox("Choose a symbol:", symbol_names)

img_path = BUILT_INS[choice]
img = load_or_warn(img_path)

if img is not None:

    st.image(img, caption=f"{choice} (source)", use_column_width=True)
    arr = np.array(img).astype(np.float32)

    # ======================================================
    # 2 — FFT RESONANCE MAP
    # ======================================================
    st.subheader("FFT Resonance Map")

    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1)

    st.image(safe_image(mag), caption="Resonance Spectrum", use_column_width=True)

    # ======================================================
    # 3 — RESONANCE ACTIVATION
    # ======================================================
    st.subheader("Activation Frequency")

    freq = st.slider("Activation frequency (Hz)", 1, 100, 33)

    y, x = np.indices(arr.shape)
    wave = np.sin(2 * np.pi * freq * x / arr.shape[1])
    resonance = (arr / 255.0) * (wave + 1)

    st.image(safe_image(resonance), caption=f"Response @ {freq} Hz", use_column_width=True)

    # ======================================================
    # 4 — HOLOGRAPHIC GEOMETRY CLASSIFIER
    # ======================================================
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

    # Rotational harmonic signature
    fft1d = np.abs(np.fft.fft(hist))
    rot_harmonics = (fft1d > 0.2 * fft1d.max()).sum()

    # Geometry code
    geo_code = f"G{lobe_count}-V{v_score:.2f}-H{h_score:.2f}-R{rot_harmonics}"
    st.subheader("Geometry Code:")
    st.code(geo_code)

# ======================================================
# 5 — MULTI-SYMBOL HARMONIC COMPOSER
# ======================================================
st.header("Multi-Symbol Harmonic Composer")

multi_files = st.file_uploader(
    "Upload 2–3 symbols to combine:",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if multi_files:
    imgs = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in multi_files]
    shapes = set([im.shape for im in imgs])

    if len(shapes) > 1:
        st.error("All images must be the same size.")
    else:
        stacked = np.stack(imgs, axis=0)

        st.subheader("Set Harmonic Weights")
        weights = []
        for i, f in enumerate(multi_files):
            w = st.slider(f"Weight for {f.name}", 0.0, 3.0, 1.0, 0.1)
            weights.append(w)

        weights = np.array(weights).reshape(-1,1,1)
        combined = np.sum(stacked * weights, axis=0)

        st.subheader("Composite Harmonic Output")
        st.image(safe_image(combined), use_column_width=True)

# ======================================================
# 6 — SEQUENCE ENGINE (A → B → C)
# ======================================================
st.header("Sequence Engine")

seq_files = st.file_uploader(
    "Upload up to 3 symbols in sequence order:",
    type=["jpg","jpeg","png"],
    accept_multiple_files=True
)

if seq_files:
    imgs = [np.array(Image.open(f).convert("L"), dtype=np.float32) for f in seq_files]

    if len(set([im.shape for im in imgs])) > 1:
        st.error("All images must be the same size.")
    else:
        st.write("Sequence:", " → ".join([f.name for f in seq_files]))

        out = imgs[0] / 255.0
        for next_img in imgs[1:]:
            out = np.sin(out + next_img/255.0)

        st.subheader("Sequence Output")
        st.image(safe_image(out), use_column_width=True)
