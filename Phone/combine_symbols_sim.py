import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# -----------------------------------------------
# --- Utility Functions (kept minimal) ---
# -----------------------------------------------

def load_image_as_heightmap(img: Image.Image, size=512):
    img = img.convert("L").resize((size, size))
    arr = np.array(img).astype(np.float32)
    height = 1.0 - (arr / 255.0)   # black = 1.0, white = 0.0
    return height

def fresnel_phase(height, alpha=2*np.pi):
    return np.exp(1j * alpha * height)

def fresnel_propagate(field, wavelength=532e-9, z=0.20, dx=10e-6):
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2) * (2*np.pi)**2)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F * H)))

def combine_fields(fields, weights, phases_deg):
    combined = np.zeros_like(fields[0], dtype=np.complex128)
    for f, w, p in zip(fields, weights, phases_deg):
        combined += w * f * np.exp(1j * np.deg2rad(p))
    intensity = np.abs(combined)**2
    return combined, intensity


# -----------------------------------------------
# --- Streamlit Session Initialization ---
# -----------------------------------------------

if "symbols" not in st.session_state:
    st.session_state.symbols = {}   # name → heightmap


st.title("🔱 Indus Symbol Multi-Symbol Resonance Simulator")
st.write("Upload symbols one at a time. They’ll be stored in memory for combination experiments.")


# -----------------------------------------------
# --- Upload Section ---
# -----------------------------------------------

uploaded = st.file_uploader("Upload a symbol PNG / JPG", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    height = load_image_as_heightmap(img)
    name = st.text_input("Symbol name", value=uploaded.name)

    if st.button("Save symbol to library"):
        st.session_state.symbols[name] = height
        st.success(f"Saved: {name}")


st.write("### 📚 Stored Symbols")
if len(st.session_state.symbols) == 0:
    st.info("No symbols uploaded yet.")
else:
    st.write(list(st.session_state.symbols.keys()))


# -----------------------------------------------
# --- Combination Section ---
# -----------------------------------------------

st.write("## 🔮 Multi-Symbol Combination Simulator")

if len(st.session_state.symbols) < 2:
    st.info("Upload at least 2 symbols to combine.")
    st.stop()

# Select symbols
chosen = st.multiselect("Choose symbols to combine", list(st.session_state.symbols.keys()))

if len(chosen) < 2:
    st.stop()

# Weight and phase controls
weights = []
phases = []

for name in chosen:
    col1, col2 = st.columns(2)
    with col1:
        w = st.slider(f"Weight: {name}", 0.0, 2.0, 1.0, 0.1)
    with col2:
        p = st.slider(f"Phase (deg): {name}", 0, 180, 0, 5)
    weights.append(w)
    phases.append(p)

# Run combination
if st.button("▶ Run Multi-Symbol Combination"):
    st.write("### Combining symbols...")

    # Generate complex fields
    fields = []
    for name in chosen:
        h = st.session_state.symbols[name]
        field = fresnel_phase(h)
        U = fresnel_propagate(field)
        fields.append(U)

    combined_complex, combined_intensity = combine_fields(fields, weights, phases)

    # Normalize for display
    norm_img = combined_intensity / combined_intensity.max()

    st.write("### 🌌 Combined Holographic Intensity")
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(norm_img, cmap="magma")
    ax.axis("off")
    st.pyplot(fig)

    # Download combined image
    buf = io.BytesIO()
    plt.imsave(buf, norm_img, cmap="magma")
    st.download_button(
        "⬇ Download Combined Intensity Image",
        buf.getvalue(),
        file_name="combined_intensity.png",
        mime="image/png"
    )

    # Also output numpy data
    st.download_button(
        "⬇ Download Combined Field (complex .npy)",
        data=combined_complex.astype(np.complex128).tobytes(),
        file_name="combined_complex.npy",
        mime="application/octet-stream"
    )
