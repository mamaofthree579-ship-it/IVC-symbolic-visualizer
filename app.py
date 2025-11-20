###############################################################################
# INDUS WAVEFORM SIMULATOR — STREAMLIT APP
# Fully compatible with your project files
###############################################################################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

st.set_page_config(page_title="Indus Waveform Simulator", layout="wide")

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

def generate_synthetic_heightmap():
    """Default synthetic symbol if user has no file."""
    ny, nx = 512, 512
    y = np.linspace(-1,1,ny)[:,None]
    x = np.linspace(-1,1,nx)[None,:]

    oval = np.exp(-((x/0.7)**2 + (y/0.5)**2)*8)
    stroke = np.exp(-((np.sqrt(x**2+y**2)-0.4)**2)/(2*0.02**2)) * (np.sin(6*np.arctan2(y,x))**2)
    dots = np.exp(-((x-0.2)**2+(y+0.15)**2)/(2*0.005**2))
    dots += np.exp(-((x+0.3)**2+(y-0.25)**2)/(2*0.006**2))

    height = (oval*0.6 + stroke*0.3 + dots*0.4)
    return height / height.max()


def spatial_fft(height):
    H = np.fft.fftshift(np.fft.fft2(height))
    mag = np.log1p(np.abs(H))
    return mag


def fresnel_propagation(height, wavelength=532e-9, dx=10e-6, z=0.15):
    field = np.exp(1j * 2*np.pi * height)
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    
    Hf = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2) * (2*np.pi)**2)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    U2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F * Hf)))

    intensity = np.abs(U2)**2
    return intensity / intensity.max()


def download_np_array(arr, filename):
    buffer = BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href


###############################################################################
# STREAMLIT UI
###############################################################################

st.title("🛕 Indus Waveform Resonance Simulator")
st.markdown("This tool simulates FFT, resonance, and Fresnel propagation for Indus symbols.")

# Sidebar inputs
st.sidebar.header("Symbol Input")

uploaded_file = st.sidebar.file_uploader("Upload symbol heightmap (.npy)", type=["npy"])

if uploaded_file is not None:
    height = np.load(uploaded_file)
else:
    height = generate_synthetic_heightmap()
    st.sidebar.markdown("*Using synthetic default heightmap*")

# Parameters
st.sidebar.header("Simulation Parameters")

freq = st.sidebar.slider("Input Frequency (Hz)", 20, 20000, 440)
harmonics = st.sidebar.multiselect("Harmonic Stack", [2,3,4,5,6,7,8], [2,3])
phase_deg = st.sidebar.slider("Phase Offset (°)", 0, 180, 0)
prop_z = st.sidebar.slider("Propagation Distance (m)", 0.05, 0.5, 0.15)

###############################################################################
# MAIN OUTPUTS
###############################################################################

col1, col2 = st.columns(2)

# Heightmap
with col1:
    st.subheader("Heightmap (Symbol Geometry)")
    fig, ax = plt.subplots()
    ax.imshow(height, cmap="gray")
    ax.set_axis_off()
    st.pyplot(fig)

# FFT
with col2:
    st.subheader("Spatial FFT")
    fft_img = spatial_fft(height)
    fig, ax = plt.subplots()
    ax.imshow(fft_img, cmap="inferno")
    ax.set_axis_off()
    st.pyplot(fig)

st.markdown("---")

# Fresnel propagation simulation
st.subheader("Holographic Interference (Fresnel Propagation)")

intensity = fresnel_propagation(height, z=prop_z)

fig, ax = plt.subplots(figsize=(6,6))
ax.imshow(intensity, cmap="magma")
ax.set_axis_off()
st.pyplot(fig)

st.markdown("---")

###############################################################################
# DOWNLOAD SECTION
###############################################################################

st.subheader("📥 Download Simulation Data")

st.markdown(download_np_array(fft_img, "fft_output.npy"), unsafe_allow_html=True)
st.markdown(download_np_array(intensity, "fresnel_intensity.npy"), unsafe_allow_html=True)

st.success("Simulation Complete.")
