# app.py
# Indus Resonance Lab — Full Suite + Cosmic Detector + DNA + Ritual Simulator
# Paste into app.py and run with: streamlit run app.py

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io, os, json, math
from typing import List, Dict, Any

st.set_page_config(page_title="Indus Resonance Lab — Ultimate", layout="wide")
st.title("Indus Symbol Resonance Lab — Ultimate Suite")

# ---------------------------
# Configuration / workspace sheet
# ---------------------------
WORKSPACE_SHEET_PATH = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"
PERSIST_DIR = "indus_state"
os.makedirs(PERSIST_DIR, exist_ok=True)

# ---------------------------
# Utilities: autocrop, load, safe
# ---------------------------
def autocrop(img: Image.Image, tol: int = 10) -> Image.Image:
    gray = img.convert("L")
    arr = np.array(gray)
    mask = arr < (255 - tol)
    if mask.any():
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img.crop((x0, y0, x1, y1))
    return img

def load_and_normalize(img_file, target_size:int=256) -> np.ndarray:
    if isinstance(img_file, str):
        img = Image.open(img_file).convert("L")
    else:
        img_file.seek(0)
        img = Image.open(img_file).convert("L")
    img = autocrop(img)
    w, h = img.size
    scale = target_size / max(w, h)
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    canvas = Image.new("L", (target_size, target_size), 255)
    offset = ((target_size - img.size[0])//2, (target_size - img.size[1])//2)
    canvas.paste(img, offset)
    arr = np.array(canvas).astype(np.float32)
    if arr.max() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    return arr

def safe_image(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return a

# ---------------------------
# Geometry extraction feature set (return small dict)
# ---------------------------
def geometry_features_from_image(arr: np.ndarray) -> Dict[str,Any]:
    N = arr.shape[0]
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1.0)
    norm = mag / (mag.max() + 1e-12)

    # symmetry
    v_sym = np.sum(np.abs(norm - np.fliplr(norm)))
    v_score = float(max(0.0, 1 - (v_sym / norm.size)))
    h_sym = np.sum(np.abs(norm - np.flipud(norm)))
    h_score = float(max(0.0, 1 - (h_sym / norm.size)))

    # radial lobes (angular histogram)
    cy, cx = N//2, N//2
    yy, xx = np.indices((N,N))
    angles = np.arctan2(yy - cy, xx - cx)
    bins = np.linspace(-np.pi, np.pi, 36)
    hist, _ = np.histogram(angles, bins=bins, weights=norm)
    lobe_count = int((hist > (0.25 * (hist.max() if hist.max()>0 else 1))).sum()) if hist.max()>0 else 0

    # rotational harmonics
    fft1d = np.abs(np.fft.fft(hist))
    rot_harmonics = int((fft1d > (0.2 * (fft1d.max() if fft1d.max()>0 else 1))).sum()) if fft1d.max()>0 else 0

    # centroid shift
    M = norm.sum() + 1e-12
    cyc = np.sum(norm * yy) / M
    cxc = np.sum(norm * xx) / M
    centroid_shift = float(np.sqrt((cxc - cx)**2 + (cyc - cy)**2) / (N/2.0))

    # roughness (Laplacian variance)
    # small numeric approximation of second derivative energy
    lap = np.abs(np.fft.ifft2((np.fft.fft2(arr) * (-4 * (np.sin(np.pi * xx/N)**2 + np.sin(np.pi * yy/N)**2)))))
    roughness = float(np.var(np.abs(lap)))

    # derive a simple band-energy fingerprint (low/mid/high)
    low_band = np.mean(np.abs(np.fft.fftshift(np.fft.fft2(arr))[0: N//8, :]))
    mid_band = np.mean(np.abs(np.fft.fftshift(np.fft.fft2(arr))[N//8: N//3, :]))
    high_band = np.mean(np.abs(np.fft.fftshift(np.fft.fft2(arr))[N//3:, :]))

    return {
        "lobe_count": lobe_count,
        "v_score": round(v_score,4),
        "h_score": round(h_score,4),
        "rot_harmonics": rot_harmonics,
        "centroid_shift": round(centroid_shift,4),
        "roughness": round(roughness,8),
        "low_band": float(low_band),
        "mid_band": float(mid_band),
        "high_band": float(high_band),
        "mag": mag  # keep for visualizations
    }

def geometry_code(features: Dict[str,Any]) -> str:
    return f"G{features['lobe_count']}-V{features['v_score']:.2f}-H{features['h_score']:.2f}-R{features['rot_harmonics']}"

# ---------------------------
# Cosmic Geometry Detector (circles / ring detection via radial profile)
# ---------------------------
def detect_circles_and_rings(arr: np.ndarray, threshold_ratio=0.25) -> Dict[str,Any]:
    N = arr.shape[0]
    cy, cx = N//2, N//2
    yy, xx = np.indices(arr.shape)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_int = r.astype(np.int32)
    maxr = int(r_int.max())
    radial_means = np.zeros(maxr+1)
    counts = np.zeros(maxr+1)
    flat = arr.copy()
    for i in range(maxr+1):
        mask = r_int == i
        counts[i] = mask.sum()
        if counts[i] > 0:
            radial_means[i] = flat[mask].mean()
    # normalize radial profile
    radial = radial_means.copy()
    if radial.max() > 0:
        radial = (radial - radial.min()) / (radial.max() - radial.min() + 1e-12)
    # detect peaks in radial profile -> possible rings
    peaks = []
    for i in range(2, len(radial)-2):
        if radial[i] > radial[i-1] and radial[i] > radial[i+1] and radial[i] > threshold_ratio:
            peaks.append((i, float(radial[i])))
    # estimate circular symmetry score via variance of radial means
    radial_var = float(np.var(radial))
    circularity_score = 1.0 - (radial_var / (radial_var + 1.0))
    return {"radial_profile": radial.tolist(), "rings": peaks, "circularity_score": circularity_score}

# ---------------------------
# Symbol DNA extractor (polar sampling fingerprint)
# ---------------------------
def symbol_dna(arr: np.ndarray, num_angles=128, num_radii=64) -> np.ndarray:
    # produce polar-sampled descriptor (num_angles * num_radii -> flatten to 1D)
    N = arr.shape[0]
    cy, cx = N//2, N//2
    thetas = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
    radii = np.linspace(0, N//2, num_radii)
    descriptor = np.zeros((num_angles, num_radii))
    for i, th in enumerate(thetas):
        xs = (cx + radii * np.cos(th)).astype(np.int32)
        ys = (cy + radii * np.sin(th)).astype(np.int32)
        xs = np.clip(xs, 0, N-1); ys = np.clip(ys, 0, N-1)
        descriptor[i,:] = arr[ys, xs]
    # radial averaging to produce 256-length vector (or flatten)
    vec = descriptor.mean(axis=0)  # radial mean -> length=num_radii
    # upscale/resize to 256
    out = np.interp(np.linspace(0, num_radii-1, 256), np.arange(num_radii), vec)
    # normalize
    out = (out - out.min()) / (out.max() - out.min() + 1e-12)
    return out  # 256-length fingerprint

# ---------------------------
# Audio synth (WAV bytes) — harmonics list + base freq
# ---------------------------
def synth_tone(freq: float, duration: float=1.5, sr:int=22050, harmonics: List[float]=None, amps:List[float]=None):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    signal = np.zeros_like(t)
    if harmonics is None:
        harmonics = [1.0]
    if amps is None:
        amps = [1.0]*len(harmonics)
    for h, a in zip(harmonics, amps):
        signal += a * np.sin(2*np.pi*freq*h*t)
    sig = signal / (np.max(np.abs(signal)) + 1e-12)
    samples = np.int16(sig * 32767)
    import wave, struct
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack('<' + ('h'*len(samples)), *samples))
    buf.seek(0)
    return buf.read()

# ---------------------------
# Multi-symbol Ritual Activation Simulator
# - Optical: complex-field sum using phase offsets (scalar approx)
# - Acoustic: phasor sum across frequencies
# ---------------------------
def ritual_optical_simulator(arr_list: List[np.ndarray], weights: List[float], phase_offsets: List[float], alpha=2*np.pi):
    # treat each arr as phase mask: field = exp(i*alpha*arr)
    fields = []
    for arr in arr_list:
        field = np.exp(1j * alpha * arr)
        fields.append(field)
    combined = np.zeros_like(fields[0], dtype=np.complex128)
    for fld, w, ph in zip(fields, weights, phase_offsets):
        combined += w * fld * np.exp(1j * ph)
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(combined)))  # show interference in Fourier domain (qualitative)
    return safe_image(np.log(1 + intensity.real))

def ritual_acoustic_simulator(spectra: List[np.ndarray], freqs: np.ndarray, amplitudes: List[float], phase_offsets: List[float]):
    # spectra: list of amplitude arrays same length as freqs
    combined = np.zeros_like(spectra[0], dtype=np.complex128)
    for spec, amp, ph in zip(spectra, amplitudes, phase_offsets):
        combined += amp * spec * np.exp(1j * ph)
    return freqs, np.abs(combined)

# ---------------------------
# Small persistent session log helper
# ---------------------------
SESSION_FILE = os.path.join(PERSIST_DIR, "session_log.json")
if "session_log" not in st.session_state:
    if os.path.exists(SESSION_FILE):
        try:
            with open(SESSION_FILE, "r") as f:
                st.session_state.session_log = json.load(f)
        except Exception:
            st.session_state.session_log = []
    else:
        st.session_state.session_log = []

def record(evt:Dict[str,Any]):
    st.session_state.session_log.append(evt)
    with open(SESSION_FILE, "w") as f:
        json.dump(st.session_state.session_log, f, indent=2)

# ---------------------------
# UI: Upload / auto-split / preprocess
# ---------------------------
st.header("1) Upload / Auto-split / Preprocess")
uploaded = st.file_uploader("Upload symbol images (JPG/PNG) — multiple allowed", type=["jpg","png","jpeg"], accept_multiple_files=True)
use_sheet = st.checkbox("Auto-split workspace sheet (if present)", value=False)
processing = []

if use_sheet and os.path.exists(WORKSPACE_SHEET_PATH):
    try:
        sheet = Image.open(WORKSPACE_SHEET_PATH).convert("L")
        w,h = sheet.size
        st.write(f"Workspace sheet found: {WORKSPACE_SHEET_PATH} — size {w}×{h}")
        if st.button("Split sheet into 3 crops and add to pipeline"):
            for i in range(3):
                left = int(i * w / 3)
                right = int((i+1) * w / 3)
                crop = sheet.crop((left, 0, right, h))
                buf = io.BytesIO(); crop.save(buf, format='PNG'); buf.seek(0)
                processing.append(buf)
            st.success("3 crops added — analyze below")
    except Exception as e:
        st.warning(f"Could not auto-split sheet: {e}")

if uploaded:
    processing.extend(uploaded)

# allow session manual crops appended earlier (visual crop editor elsewhere)
if "manual_crops" in st.session_state:
    processing.extend(st.session_state.manual_crops)

if len(processing) == 0:
    st.info("Upload files or auto-split the workspace sheet to begin.")
    st.stop()

# normalize all processing items
arrs = []
names = []
for p in processing:
    try:
        arr = load_and_normalize(p, target_size=256)
        arrs.append(arr); names.append(getattr(p, "name", "uploaded"))
    except Exception as e:
        st.warning(f"Skipping one file: {e}")

# show thumbnails
st.subheader("Preprocessed symbols")
cols = st.columns(len(arrs))
for c, a, n in zip(cols, arrs, names):
    c.image(safe_image(a), caption=n, use_column_width=True)

# ---------------------------
# 2) Feature extraction + cosmic detection + DNA
# ---------------------------
st.header("2) Feature extraction • Cosmic detectors • DNA fingerprints")

features_list = []
dna_list = []
cosmic_list = []

for i, arr in enumerate(arrs):
    feats = geometry_features_from_image(arr)
    dna = symbol_dna(arr, num_angles=128, num_radii=64)
    cosmic = detect_circles_and_rings(arr)
    features_list.append(feats); dna_list.append(dna); cosmic_list.append(cosmic)

    st.markdown(f"---\n### Symbol {i+1} — {names[i]}")
    st.write("Geometry code:", geometry_code(feats))
    st.write("Key features:", {k:feats[k] for k in ["lobe_count","v_score","h_score","rot_harmonics","centroid_shift","roughness"]})
    st.write("Cosmic detector: circularity score:", round(cosmic["circularity_score"],3))
    if len(cosmic["rings"])>0:
        st.write("Detected radial peaks (rings):", cosmic["rings"][:5])
    st.write("DNA fingerprint (256-length):")
    st.line_chart(dna)
    record({"type":"analyze","name":names[i],"features":feats,"cosmic": {"circularity":cosmic["circularity_score"], "rings":len(cosmic["rings"])}})

# ---------------------------
# 3) Meaning inference (layered)
# ---------------------------
st.header("3) Meaning inference (layered)")

for i, feats in enumerate(features_list):
    # basic band inference
    low = feats["low_band"]; mid = feats["mid_band"]; hi = feats["high_band"]
    meaning = []
    if low > mid and low > hi:
        meaning.append("Material / Trade / Goods")
    if mid > low and mid > hi:
        meaning.append("Civic / Craft / Coordination")
    if hi > low and hi > mid:
        meaning.append("Ritual / Abstract / Conceptual")
    if feats["v_score"] > 0.7:
        meaning.append("Authority / Order")
    if feats["h_score"] > 0.7:
        meaning.append("Cooperation / Pairing")
    # cosmic
    if cosmic_list[i]["circularity_score"] > 0.6:
        meaning.append("Cosmic / Astral mapping (circularity)")
    st.markdown(f"**Symbol {i+1} — {names[i]} inferred layers:**")
    for m in meaning:
        st.write("- ", m)

# ---------------------------
# 4) Audio mapping & playback (DNA -> freq mapping)
# ---------------------------
st.header("4) Audio activation from DNA")

def dna_to_freq_and_harmonics(dna_vec: np.ndarray, base_min=30, base_max=800, n_harm=4):
    # Map DNA's spectral centroid to a base frequency between base_min..base_max
    idx = np.arange(len(dna_vec))
    centroid = np.sum(idx * dna_vec) / (dna_vec.sum() + 1e-12)
    frac = centroid / (len(dna_vec)-1)
    base = base_min + frac * (base_max - base_min)
    # harmonic multipliers (1..n_harm)
    harm_mults = [i+1 for i in range(n_harm)]
    harm_amps = [1.0/(i+1) for i in range(n_harm)]
    return float(base), harm_mults, harm_amps

st.write("Select a symbol to synthesize from its DNA")
sel = st.selectbox("Pick symbol", options=list(range(len(arrs))), format_func=lambda x: names[x])
if sel is not None:
    n_harm = st.slider("Number of harmonics", 1, 8, 4)
    base, harms, amps = dna_to_freq_and_harmonics(dna_list[sel], base_min=40, base_max=800, n_harm=n_harm)
    st.write(f"Base freq suggested: {int(base)} Hz")
    if st.button("Play DNA-derived tone"):
        wav = synth_tone(base, duration=2.0, harmonics=harms, amps=amps)
        st.audio(wav, format="audio/wav")
        record({"type":"audio_play","symbol":names[sel],"base_freq":base,"harmonics":harms})

# ---------------------------
# 5) Multi-symbol Ritual activation simulator (optical + acoustic)
# ---------------------------
st.header("5) Ritual Activation Simulator — Multi-symbol")

if len(arrs) >= 1:
    st.write("Choose symbols (phase offsets in degrees, weights). If you choose 1 symbol it's trivially that symbol.")
    picks = st.multiselect("Pick symbol indices to include", options=list(range(len(arrs))), default=list(range(len(arrs))))
    if len(picks) == 0:
        st.info("Pick at least one symbol")
    else:
        weights = []
        phases_deg = []
        for idx in picks:
            w = st.slider(f"Weight for {names[idx]}", 0.0, 3.0, 1.0, key=f"w_{idx}")
            ph = st.slider(f"Phase (deg) for {names[idx]}", 0, 360, int(90*idx)%360, key=f"ph_{idx}")
            weights.append(float(w)); phases_deg.append(math.radians(float(ph)))
        # run optical simulator
        opt_img = ritual_optical_simulator([arrs[i] for i in picks], weights, phases_deg, alpha=4*np.pi)  # alpha tuned
        st.subheader("Optical Interference (simulated intensity)")
        st.image(opt_img, use_column_width=True)
        # acoustic: build simple Gaussian spectra centered on DNA-mapped base freq
        freqs = np.linspace(1,1000,1000)
        spectra = []
        amps = []
        for i in picks:
            base, harm_mults, harm_amps = dna_to_freq_and_harmonics(dna_list[i], base_min=30, base_max=800, n_harm=3)
            spec = np.zeros_like(freqs)
            for mult, a in zip(harm_mults, harm_amps):
                peak = base * mult
                spec += a * np.exp(-0.5*((freqs - peak)/(peak*0.05+1e-6))**2)
            spectra.append(spec)
            amps.append(1.0)
        # acoustic phase offsets reuse phases_deg
        freqs_out, combined_spec = ritual_acoustic_simulator(spectra, freqs, amps, phases_deg)
        st.subheader("Acoustic interference (magnitude)")
        # show reduced range plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,2))
        ax.plot(freqs_out, combined_spec)
        ax.set_xlim(0, 400)
        ax.set_xlabel("Hz"); ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        # allow playback of dominant peak
        peak_idx = np.argmax(combined_spec)
        peak_freq = freqs_out[peak_idx]
        st.write(f"Dominant combined peak: {int(peak_freq)} Hz")
        if st.button("Play combined dominant tone"):
            wav = synth_tone(float(peak_freq), duration=2.0, harmonics=[1,2,3], amps=[1.0,0.3,0.15])
            st.audio(wav, format="audio/wav")
            record({"type":"ritual_play","peak_freq":float(peak_freq),"picks":picks})

# ---------------------------
# 6) Exports, clustering, and session log
# ---------------------------
st.header("6) Clustering, Export & Session Log")
if st.button("Run k-means on DNA fingerprints (k=3)"):
    X = np.vstack(dna_list)
    # reduce dimensionality via simple average pooling to speed clustering
    Xr = X.reshape(len(X), 64, 4).mean(axis=2)
    # simple kmeans np implementation
    k = min(3, len(Xr))
    # init random centroids
    rng = np.random.RandomState(1)
    centroids = Xr[rng.choice(len(Xr), k, replace=False)]
    for _ in range(100):
        dists = np.linalg.norm(Xr[:,None,:] - centroids[None,:,:], axis=2)
        labels = np.argmin(dists, axis=1)
        newc = np.array([Xr[labels==i].mean(axis=0) if np.any(labels==i) else centroids[i] for i in range(k)])
        if np.allclose(newc, centroids):
            break
        centroids = newc
    st.write("Cluster labels:")
    for i, lab in enumerate(labels):
        st.write(f"- {names[i]} -> cluster {int(lab)}")
    record({"type":"cluster","labels":labels.tolist()})

if st.button("Download session log (JSON)"):
    b = json.dumps(st.session_state.session_log, indent=2).encode('utf-8')
    st.download_button("Download log", b, file_name="indus_session_log.json", mime="application/json")

st.write("Session events (last 20):")
st.write(st.session_state.session_log[-20:])
