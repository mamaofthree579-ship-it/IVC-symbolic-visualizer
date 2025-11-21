import streamlit as st
import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
import io
from sklearn.cluster import KMeans

st.set_page_config(page_title="Indus Resonance Lab v3", layout="wide")

# =======================================================
# AUTO-CROP + NORMALIZE
# =======================================================

def autocrop(img, tol=10):
    gray = img.convert("L")
    arr = np.array(gray)
    mask = arr < (255 - tol)
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img.crop((x0, y0, x1, y1))
    return img

def load_normalized(img_file, size=256):
    img = Image.open(img_file).convert("L")
    img = autocrop(img)

    # Resize longest side
    w, h = img.size
    scale = size / max(w, h)
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    # Pad square
    canvas = Image.new("L", (size, size), 255)
    off = ((size - img.size[0])//2, (size - img.size[1])//2)
    canvas.paste(img, off)

    arr = np.array(canvas).astype(np.float32)
    if arr.max() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr

def safe(arr):
    arr = np.nan_to_num(arr)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr

# =======================================================
# MEANING INFERENCE ENGINE — EXPANDED
# =======================================================

def meaning_engine(arr):
    fft_mag = np.abs(fftshift(fft2(arr)))

    low = np.mean(fft_mag[0:40])
    mid = np.mean(fft_mag[80:140])
    hi  = np.mean(fft_mag[180:250])

    # symmetry
    vert = np.mean(np.abs(arr - np.fliplr(arr)))
    horiz = np.mean(np.abs(arr - np.flipud(arr)))

    meanings = []
    # Basic bands
    if low > mid and low > hi:
        meanings.append("Material / Trade / Goods")
    if mid > low and mid > hi:
        meanings.append("Civic / Craft / Coordination")
    if hi > low and hi > mid:
        meanings.append("Ritual / Abstract / Conceptual")

    # Symmetry reflections
    if vert < 0.08:
        meanings.append("Authority / Order / Hierarchy (Vertical Symmetry)")
    if horiz < 0.08:
        meanings.append("Duality / Cooperation / Pairing (Horizontal Symmetry)")

    # Higher-level geometry
    radial_balance = abs(low - hi) < 0.03
    if radial_balance:
        meanings.append("Cosmic Mapping / Cyclic Knowledge (Radial Balanced)")

    # Layer 3: structural inference
    edges = np.sum(np.abs(np.diff(arr, axis=0))) + np.sum(np.abs(np.diff(arr, axis=1)))
    if edges > 20000:
        meanings.append("High Structural Density → Administrative or Legal")
    elif edges < 7000:
        meanings.append("Low Structural Density → Symbolic, Mythic, Emblematic")

    return meanings, fft_mag

# =======================================================
# HARMONIC COMPOSER (SAFE SLIDERS)
# =======================================================

def harmonic_combine(a, b, f1, f2, phase):
    shift = int(phase * 10)
    return safe(f1*a + f2*np.roll(b, shift, axis=1))

# =======================================================
# SEQUENCE ENGINE (SAFE IF 1+ SYMBOLS)
# =======================================================

def sequence_resonance(images, phase):
    acc = np.zeros_like(images[0])
    for i, img in enumerate(images):
        shift = int(i * phase * 10)
        acc += np.roll(img, shift, axis=0)
    return safe(acc)

# =======================================================
# CLUSTERING ENGINE (ALL SYMBOLS)
# =======================================================

def cluster_symbols(images):
    flattened = [img.flatten() for img in images]
    km = KMeans(n_clusters=min(3, len(images)), n_init='auto')
    labels = km.fit_predict(flattened)
    return labels

# =======================================================
# STREAMLIT UI
# =======================================================

st.title("🜂 Indus Symbol Resonance Lab — Full Engine v3")
st.write("Upload ANY Indus symbols. The system auto-crops, normalizes, infers meaning, builds composites, sequences, and clusters resonance patterns.")

files = st.file_uploader("Upload images", type=["png", "jpg"], accept_multiple_files=True)

if not files:
    st.stop()

# Load
images = [load_normalized(f) for f in files]

# Display
cols = st.columns(len(images))
for c, img in zip(cols, images):
    c.image(safe(img), use_container_width=True)

st.markdown("---")

# =======================================================
# MEANING LAYERS
# =======================================================

st.subheader("Meaning Inference (All Layers)")

for i, img in enumerate(images):
    m, fftmap = meaning_engine(img)
    st.write(f"### Symbol {i+1}")
    for x in m:
        st.write("- ", x)
    st.image(safe(fftmap), caption="Harmonic Spectrum")

st.markdown("---")

# =======================================================
# COMPOSITE ENGINE (SAFE)
# =======================================================

if len(images) >= 2:
    st.subheader("Harmonic Composite Engine")

    # Safe default sliders
    f1 = st.slider("Strength A", 0.0, 3.0, 1.0)
    f2 = st.slider("Strength B", 0.0, 3.0, 1.0)
    phase = st.slider("Phase shift", 0.0, 1.0, 0.1)

    comp = harmonic_combine(images[0], images[1], f1, f2, phase)
    st.image(comp, caption="Composite Output", use_container_width=True)

st.markdown("---")

# =======================================================
# SEQUENCE ENGINE (SAFE)
# =======================================================

if len(images) >= 2:
    st.subheader("Sequence Resonance Engine")

    # Prevent slider equal min/max
    phase = st.slider("Phase per symbol", 0.0, 0.5, 0.1)

    seq = sequence_resonance(images, phase)
    st.image(seq, caption="Sequence Resonance", use_container_width=True)

st.markdown("---")

# =======================================================
# CLUSTERING ENGINE (HARMONIC FAMILIES)
# =======================================================

if len(images) >= 2:
    st.subheader("Resonance Clustering (Similarity Families)")
    labels = cluster_symbols(images)

    for i, (img, lab) in enumerate(zip(images, labels)):
        st.write(f"Symbol {i+1} → Cluster {lab}")
        st.image(safe(img), width=200)
