# import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import fft2, fftshift
import io

set_page_config(page_title="Indus Symbol Resonance Lab", layout="wide")

# ======================================================
# IMAGE PREPROCESSING PIPELINE
# ======================================================

def autocrop(img, tol=10):
    """
    Auto-crops whitespace or background.
    tol = threshold sensitivity.
    """
    gray = img.convert("L")
    arr = np.array(gray)

    mask = arr < (255 - tol)
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img.crop((x0, y0, x1, y1))
    else:
        return img


def load_normalized(img_file, target_size=256):
    """
    Auto-crop → resize → center-pad → normalize to float32 [0,1].
    Works for ANY uploaded JPG or PNG.
    """
    img = Image.open(img_file).convert("L")

    # Auto-crop symbol area
    img = autocrop(img)

    # Resize longest side
    w, h = img.size
    scale = target_size / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Pad to square
    pad_img = Image.new("L", (target_size, target_size), 255)
    offset = ((target_size - img.size[0]) // 2, (target_size - img.size[1]) // 2)
    pad_img.paste(img, offset)

    arr = np.array(pad_img).astype(np.float32)
    arr = (arr - arr.min()) / max(1e-6, arr.max())
    return arr


def safe_image(arr):
    """Normalizes any array into [0,1] for Streamlit."""
    arr = np.nan_to_num(arr)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


# ======================================================
# MEANING INFERENCE ENGINE
# ======================================================
"""
This uses the “harmonic semantics” model we designed:

Low-frequency dominance → material trade  
Mid-frequency dominance → craft / civic  
High-frequency dominance → ritual / conceptual  
Vertical symmetry → authority / order  
Horizontal symmetry → cooperation / plurality  
Radial symmetry → cosmic / mythic
"""

def infer_meaning(arr):
    # FFT magnitude map
    fft_mag = np.abs(fftshift(fft2(arr)))

    low = np.mean(fft_mag[0:40])
    mid = np.mean(fft_mag[80:120])
    hi  = np.mean(fft_mag[180:250])

    # symmetry checks
    vert = np.mean(np.abs(arr - np.fliplr(arr)))
    horiz = np.mean(np.abs(arr - np.flipud(arr)))

    meaning = []

    # Frequency semantic bands
    if low > mid and low > hi:
        meaning.append("Material / Trade / Goods")
    if mid > low and mid > hi:
        meaning.append("Craft / Civic / Coordination")
    if hi > low and hi > mid:
        meaning.append("Ritual / Abstract / Conceptual")

    # Symmetry semantics
    if vert < 0.08:
        meaning.append("Order / Authority / Hierarchy (Vertical Symmetry)")

    if horiz < 0.08:
        meaning.append("Cooperation / Duality / Pairing (Horizontal Symmetry)")

    if abs(low - hi) < 0.02:
        meaning.append("Cosmic Mapping / Cyclic Knowledge (Radial-like balance)")

    return meaning, fft_mag


# ======================================================
# HARMONIC COMPOSER
# ======================================================

def harmonic_combine(arr1, arr2, f1=1.0, f2=1.0, phase=0.0):
    return safe_image(
        f1 * arr1 + f2 * np.roll(arr2, int(phase * 10), axis=1)
    )


# ======================================================
# SYMBOL SEQUENCE ENGINE
# ======================================================

def sequence_resonance(images, phase_shift=0.1):
    acc = np.zeros_like(images[0])
    for i, img in enumerate(images):
        acc += np.roll(img, int(i * phase_shift * 10), axis=0)
    return safe_image(acc)


# ======================================================
# STREAMLIT UI
# ======================================================

st.title("🜂 Indus Symbol Resonance & Meaning Inference Lab")
st.write("Upload any Indus symbols (JPG/PNG). The system **auto-crops**, **normalizes**, generates **harmonic spectra**, and infers meaning from wave-pattern semantics.")

uploaded = st.file_uploader("Upload symbol images", type=["jpg", "png"], accept_multiple_files=True)

if uploaded:
    st.subheader("Preprocessed Symbols")

    images = [load_normalized(f) for f in uploaded]

    cols = st.columns(len(images))
    for c, img in zip(cols, images):
        c.image(safe_image(img), caption="Normalized Symbol", use_column_width=True)

    st.markdown("---")

    # ------------------ Meaning Inference ------------------
    st.subheader("Meaning Inference Engine")

    for i, img in enumerate(images):
        meaning, fft_mag = infer_meaning(img)

        st.write(f"### Symbol {i+1}")
        st.image(safe_image(img), width=200)
        st.write("**Inferred Meaning Layers:**")
        for m in meaning:
            st.write(f"- {m}")

        st.write("**Harmonic Spectrum:**")
        st.image(safe_image(fft_mag), use_column_width=True)
        st.markdown("---")

    # ------------------ Harmonic Composer ------------------
    if len(images) >= 2:
        st.subheader("Harmonic Composite Generator")

        f1 = st.slider("Strength A", 0.0, 3.0, 1.0)
        f2 = st.slider("Strength B", 0.0, 3.0, 1.0)
        phase = st.slider("Phase shift", 0.0, 1.0, 0.1)

        composite = harmonic_combine(images[0], images[1], f1=f1, f2=f2, phase=phase)

        st.image(composite, caption="Composite Output", use_column_width=True)

    # ------------------ Sequence Engine ------------------
    if len(images) >= 3:
        st.subheader("Sequence Resonance Engine")

        phase = st.slider("Phase shift per symbol", 0.0, 0.5, 0.1)
        seq = sequence_resonance(images, phase_shift=phase)

        st.image(seq, caption="Sequence Resonance Output", use_column_width=True)
