# app_extended.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple

st.set_page_config(page_title="Indus Multi-Combiner (Optical + Acoustic)", layout="wide")

# ---------------------------------------------------------------------
# Presets: activation-theory defaults (scaled for 0-100Hz slider by default)
# ---------------------------------------------------------------------
PRESETS = {
    "Jar": {
        "label": "Jar",
        "base_freq": 20.0,    # scaled baseline
        "harmonics": [40.0, 60.0],
        "amplitude": 1.0,
        "phase_deg": 0.0
    },
    "Fish": {
        "label": "Fish",
        "base_freq": 48.0,
        "harmonics": [96.0, 144.0],
        "amplitude": 0.9,
        "phase_deg": 45.0
    },
    "Double-Fish": {
        "label": "Double-Fish",
        "base_freq": 88.0,
        "harmonics": [176.0, 264.0],
        "amplitude": 1.1,
        "phase_deg": 90.0
    },
    "U-Stand": {
        "label": "U-Stand",
        "base_freq": 60.0,
        "harmonics": [120.0, 180.0],
        "amplitude": 1.0,
        "phase_deg": 0.0
    },
    "Fanged-Fish": {
        "label": "Fanged-Fish",
        "base_freq": 72.0,
        "harmonics": [144.0, 216.0],
        "amplitude": 1.0,
        "phase_deg": 30.0
    }
}

# ---------------------------------------------------------------------
# Utilities: image -> heightmap, optical Fresnel, synthetic acoustic
# ---------------------------------------------------------------------
def load_image_as_heightmap(img: Image.Image, size=512):
    img = img.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    height = 1.0 - (arr / 255.0)  # black -> raised (1.0)
    return height

def fresnel_phase_field(height, alpha=2*np.pi):
    return np.exp(1j * alpha * height)

def fresnel_propagate(field, wavelength=532e-9, z=0.20, dx=10e-6):
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2) * (2*np.pi)**2)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    U = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return U

def spatial_fft(height):
    H = np.fft.fftshift(np.fft.fft2(height))
    return np.log1p(np.abs(H))

# Map dominant spatial frequency of shape to a synthetic acoustic peak (toy model)
def estimate_spatial_freq_to_audio_peak(height, dx_pixels=1.0, scale_map=100.0):
    # compute 2D FFT magnitude and find centroid of spectral energy
    H = np.fft.fftshift(np.fft.fft2(height))
    mag = np.abs(H)
    ny, nx = mag.shape
    # frequency axes (spatial)
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dx_pixels))
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx_pixels))
    FY, FX = np.meshgrid(fy, fx, indexing='ij')
    # spectral centroid (radial)
    radial = np.sqrt(FX**2 + FY**2)
    centroid = np.sum(radial * mag) / (np.sum(mag) + 1e-12)
    # map centroid to an audio frequency via scale_map (tunable)
    base_freq = float(np.abs(centroid) * scale_map + 10.0)  # ensure > 0
    return base_freq

def gaussian_spectrum(freqs: np.ndarray, peak_freq: float, Q=10.0, amp=1.0):
    # produce a Gaussian-ish spectral peak (sigma related to Q)
    sigma = peak_freq / (Q + 1e-12)
    return amp * np.exp(-0.5 * ((freqs - peak_freq) / sigma) ** 2)

# Combine acoustic spectra as complex phasors with phase offsets (deg)
def combine_acoustic_spectra(spectra_list: List[np.ndarray],
                             freqs: np.ndarray,
                             phase_offsets_deg: List[float],
                             weights: List[float]):
    combined = np.zeros_like(spectra_list[0], dtype=np.complex128)
    for spec, ph_deg, w in zip(spectra_list, phase_offsets_deg, weights):
        combined += w * spec * np.exp(1j * np.deg2rad(ph_deg))
    return combined, np.abs(combined)

# Helper to auto-split a 3-icon image into three vertical crops
def split_three_horizontal(path: str, size=512) -> List[np.ndarray]:
    img = Image.open(path).convert("L")
    w, h = img.size
    # split into 3 equal vertical slices (left|center|right)
    xs = [0, w//3, 2*w//3, w]
    parts = []
    for i in range(3):
        crop = img.crop((xs[i], 0, xs[i+1], h))
        hm = load_image_as_heightmap(crop, size=size)
        parts.append(hm)
    return parts

# ---------------------------------------------------------------------
# Session state: stored symbols & optional measured spectra
# ---------------------------------------------------------------------
if "symbols" not in st.session_state:
    st.session_state.symbols = {}  # name -> dict(heightmap, preset, measured_spec_df)

st.title("🔱 Indus Multi-Combiner — Optical + Acoustic + Presets")
st.markdown("Upload symbols one-by-one. Use presets to auto-populate activation parameters. Combine optical fields and acoustic spectra together.")

# -- Auto-load sample 3-symbol sheet (developer-provided path) --
st.info("If you previously uploaded a 3-symbol sheet, you can auto-split it into 3 symbols.")
if st.button("Auto-load sample 3-symbol sheet (workspace image)"):
    sample_path = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"
    try:
        parts = split_three_horizontal(sample_path, size=512)
        names = ["sample_A", "sample_B", "sample_C"]
        for n, hm in zip(names, parts):
            st.session_state.symbols[n] = {
                "height": hm,
                "preset": None,
                "measured_spec": None
            }
        st.success("Loaded sample_A, sample_B, sample_C from workspace image.")
    except Exception as e:
        st.error(f"Could not load sample sheet: {e}")

# -- Upload single symbol --
uploaded = st.file_uploader("Upload symbol PNG / JPG (one at a time)", type=["png", "jpg", "jpeg"])
if uploaded:
    try:
        img = Image.open(uploaded)
        name = st.text_input("Enter name for this symbol", value=uploaded.name)
        if st.button("Save uploaded symbol to library"):
            hm = load_image_as_heightmap(img, size=512)
            st.session_state.symbols[name] = {"height": hm, "preset": None, "measured_spec": None}
            st.success(f"Saved symbol: {name}")
    except Exception as e:
        st.error(f"Upload error: {e}")

st.write("### Stored symbols")
if len(st.session_state.symbols) == 0:
    st.info("No symbols in library. Upload or auto-load sample sheet.")
else:
    cols = st.columns([2,1,1])
    cols[0].write("Name")
    cols[1].write("Preview")
    cols[2].write("Controls")
    for nm, data in st.session_state.symbols.items():
        row_cols = st.columns([2,1,1])
        row_cols[0].write(nm)
        # preview small
        fig, ax = plt.subplots(figsize=(2,2))
        ax.imshow(data["height"], cmap="gray")
        ax.axis("off")
        row_cols[1].pyplot(fig)
        if row_cols[2].button(f"Remove: {nm}"):
            st.session_state.symbols.pop(nm, None)
            st.experimental_rerun()

# ---------------------------------------------------------------------
# Selection of symbols to combine
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Select symbols & apply presets")
chosen = st.multiselect("Symbols to combine (choose 2+)", list(st.session_state.symbols.keys()))

if len(chosen) >= 2:
    # UI for presets per chosen symbol
    st.write("## Activation presets (choose preset or manual values)")
    # Build controls
    controls = {}
    for name in chosen:
        block = st.expander(f"Activation for {name}", expanded=False)
        with block:
            preset_choice = st.selectbox(f"Preset for {name}", options=["(none)"] + list(PRESETS.keys()), key=f"preset_{name}")
            if preset_choice != "(none)":
                p = PRESETS[preset_choice]
                base = st.number_input(f"Base freq (Hz) for {name}", value=float(p["base_freq"]), key=f"base_{name}")
                amp = st.number_input(f"Amplitude for {name}", min_value=0.0, value=float(p["amplitude"]), key=f"amp_{name}")
                phase = st.slider(f"Phase (deg) for {name}", 0, 360, value=int(p["phase_deg"]), key=f"phase_{name}")
            else:
                # manual entry
                base = st.number_input(f"Base freq (Hz) for {name}", value=20.0, key=f"base_{name}")
                amp = st.number_input(f"Amplitude for {name}", min_value=0.0, value=1.0, key=f"amp_{name}")
                phase = st.slider(f"Phase (deg) for {name}", 0, 360, value=0, key=f"phase_{name}")

            # Optionally upload measured acoustic CSV for this symbol
            spec_upload = st.file_uploader(f"Upload measured spectrum CSV for {name} (freq,amp)", key=f"spec_up_{name}")
            measured_df = None
            if spec_upload:
                try:
                    measured_df = pd.read_csv(spec_upload)
                    st.write("Preview of uploaded measured spectrum:")
                    st.line_chart(measured_df.iloc[:,1].values, width=300)
                except Exception as e:
                    st.error(f"Could not read CSV: {e}")

            controls[name] = {"base": base, "amp": amp, "phase": phase, "measured_df": measured_df}
    # Frequency axis settings for acoustic combiner
    st.write("---")
    st.write("Acoustic spectrum axis (for synthetic and combination):")
    fmin = st.number_input("Min frequency (Hz)", value=1.0)
    fmax = st.number_input("Max frequency (Hz)", value=200.0)
    npts = st.number_input("Spectrum points", value=200, step=50)
    freqs = np.linspace(float(fmin), float(fmax), int(npts))

    if st.button("▶ Run optical + acoustic combination"):
        # Optical: build fields for chosen symbols and combine (existing)
        fields = []
        for name in chosen:
            hm = st.session_state.symbols[name]["height"]
            field = fresnel_phase_field(hm)
            U = fresnel_propagate(field)
            fields.append(U)

        # optical combine weights & phases (use amp -> weight, phase)
        weights_opt = [controls[n]["amp"] for n in chosen]
        phases_opt = [controls[n]["phase"] for n in chosen]
        combined_opt = np.zeros_like(fields[0], dtype=np.complex128)
        for f, w, ph in zip(fields, weights_opt, phases_opt):
            combined_opt += w * f * np.exp(1j * np.deg2rad(ph))
        combined_intensity = np.abs(combined_opt)**2
        norm_img = combined_intensity / (combined_intensity.max() + 1e-12)

        st.write("### 🌌 Combined Optical (Fresnel) Intensity")
        fig1, ax1 = plt.subplots(figsize=(5,5))
        ax1.imshow(norm_img, cmap="magma")
        ax1.axis("off")
        st.pyplot(fig1)
        # download optical image
        buf = io.BytesIO()
        plt.imsave(buf, norm_img, cmap="magma")
        st.download_button("Download combined intensity PNG", buf.getvalue(), file_name="combined_intensity.png", mime="image/png")

        # Acoustic: generate spectra per symbol (measured or synthetic)
        spectra = []
        phases = []
        weights = []
        for name in chosen:
            ctrl = controls[name]
            # measured CSV takes precedence
            if ctrl["measured_df"] is not None:
                df = ctrl["measured_df"]
                # Align measured spectrum onto freqs by interpolation
                measured_freqs = df.iloc[:,0].values
                measured_amp = df.iloc[:,1].values
                amp_interp = np.interp(freqs, measured_freqs, measured_amp, left=0.0, right=0.0)
                spec = amp_interp
            else:
                # Produce synthetic spectrum from geometry
                hm = st.session_state.symbols[name]["height"]
                peak = estimate_spatial_freq_to_audio_peak(hm, scale_map=1.0)  # base centroid
                # map centroid (small value) to audible baseline via multiplicative mapping
                # The widget provides user-specified base freq to override
                user_base = float(ctrl["base"])
                # produce gaussian at user_base
                spec = gaussian_spectrum(freqs, peak_freq=user_base, Q=8.0, amp=float(ctrl["amp"]))
            spectra.append(spec)
            phases.append(float(ctrl["phase"]))
            weights.append(float(ctrl["amp"]))

        combined_spec_complex, combined_spec_mag = combine_acoustic_spectra(spectra, freqs, phases, weights)

        st.write("### 🔊 Combined Acoustic Spectrum (magnitude)")
        fig2, ax2 = plt.subplots(figsize=(8,3))
        for i, spec in enumerate(spectra):
            ax2.plot(freqs, spec, label=f"{chosen[i]} (individual)", alpha=0.6)
        ax2.plot(freqs, combined_spec_mag, label="Combined (mag)", color='k', linewidth=2)
        ax2.set_xlim(fmin, fmax)
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude (arb.)")
        ax2.legend()
        st.pyplot(fig2)

        # Downloads for acoustic data
        # combined complex npy
        st.download_button("Download combined acoustic complex (.npy)",
                           combined_spec_complex.tobytes(),
                           file_name="combined_acoustic_complex.npy",
                           mime="application/octet-stream")
        # combined magnitude CSV
        csv_buf = io.StringIO()
        pd.DataFrame({"freq": freqs, "amplitude": combined_spec_mag}).to_csv(csv_buf, index=False)
        st.download_button("Download combined acoustic CSV", csv_buf.getvalue(), file_name="combined_acoustic.csv", mime="text/csv")

else:
    st.info("Select at least two symbols to enable combination controls.")
