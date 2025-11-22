# app.py
# Indus Resonance Lab — Complete, self-contained application
# Features:
# - Auto-crop + normalize uploads
# - Geometry extraction + FFT visualization
# - Cosmic detector (radial/ring)
# - Symbol DNA (256 fingerprint)
# - Meaning inference layers
# - Harmonic composite, sequence engine
# - Ritual simulator (optical + acoustic)
# - Audio synth and playback
# - Sweep GIF generation (optical & acoustic)
# - Export DNA + features ZIP
# - HTML hypothesis report generation
# - Session recording and basic persistence
#
# Variables used throughout:
# arrs, names, dna_list, features_list, cosmic_list
#
# Optional workspace sample file (unused if not present)
WORKSPACE_SHEET_PATH = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"

import streamlit as st
import numpy as np
from PIL import Image
import io, os, json, csv, base64, math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indus Resonance Lab — All-in-One", layout="wide")
st.title("Indus Symbol Resonance Lab — All-in-One")

# ----------------------------
# Utilities: autocrop, normalize
# ----------------------------
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
    """Load a file path or a streamlit UploadedFile and return a 0..1 float32 array target_size^2"""
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

# ----------------------------
# Geometry & features extraction
# ----------------------------
def geometry_features_from_image(arr: np.ndarray) -> dict:
    N = arr.shape[0]
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1.0)
    norm = mag / (mag.max() + 1e-12)

    v_sym = np.sum(np.abs(norm - np.fliplr(norm)))
    v_score = float(max(0.0, 1 - (v_sym / norm.size)))
    h_sym = np.sum(np.abs(norm - np.flipud(norm)))
    h_score = float(max(0.0, 1 - (h_sym / norm.size)))

    cy, cx = N//2, N//2
    yy, xx = np.indices((N,N))
    angles = np.arctan2(yy - cy, xx - cx)
    bins = np.linspace(-np.pi, np.pi, 36)
    hist, _ = np.histogram(angles, bins=bins, weights=norm)
    lobe_count = int((hist > (0.25 * (hist.max() if hist.max()>0 else 1))).sum()) if hist.max()>0 else 0

    fft1d = np.abs(np.fft.fft(hist))
    rot_harmonics = int((fft1d > (0.2 * (fft1d.max() if fft1d.max()>0 else 1))).sum()) if fft1d.max()>0 else 0

    M = norm.sum() + 1e-12
    cyc = np.sum(norm * yy) / M
    cxc = np.sum(norm * xx) / M
    centroid_shift = float(np.sqrt((cxc - cx)**2 + (cyc - cy)**2) / (N/2.0))

    lap = np.abs(np.fft.ifft2((np.fft.fft2(arr) * (-4 * (np.sin(np.pi * xx/N)**2 + np.sin(np.pi * yy/N)**2)))))
    roughness = float(np.var(np.abs(lap)))

    # band energies (coarse)
    low_band = float(np.mean(np.abs(np.fft.fftshift(np.fft.fft2(arr))[0: N//8, :])))
    mid_band = float(np.mean(np.abs(np.fft.fftshift(np.fft.fft2(arr))[N//8: N//3, :])))
    high_band = float(np.mean(np.abs(np.fft.fftshift(np.fft.fft2(arr))[N//3:, :])))

    return {
        "lobe_count": lobe_count,
        "v_score": round(v_score,4),
        "h_score": round(h_score,4),
        "rot_harmonics": rot_harmonics,
        "centroid_shift": round(centroid_shift,4),
        "roughness": round(roughness,8),
        "low_band": low_band,
        "mid_band": mid_band,
        "high_band": high_band,
        "mag": mag  # keep for visualization
    }

def geometry_code(features: dict) -> str:
    return f"G{features['lobe_count']}-V{features['v_score']:.2f}-H{features['h_score']:.2f}-R{features['rot_harmonics']}"

# ----------------------------
# Cosmic detector
# ----------------------------
def detect_circles_and_rings(arr: np.ndarray, threshold_ratio=0.25) -> dict:
    N = arr.shape[0]
    cy, cx = N//2, N//2
    yy, xx = np.indices(arr.shape)
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r_int = r.astype(np.int32)
    maxr = int(r_int.max())
    radial_means = np.zeros(maxr+1)
    counts = np.zeros(maxr+1)
    for i in range(maxr+1):
        mask = r_int == i
        counts[i] = mask.sum()
        if counts[i] > 0:
            radial_means[i] = arr[mask].mean()
    radial = radial_means.copy()
    if radial.max() > 0:
        radial = (radial - radial.min()) / (radial.max() - radial.min() + 1e-12)
    peaks = []
    for i in range(2, len(radial)-2):
        if radial[i] > radial[i-1] and radial[i] > radial[i+1] and radial[i] > threshold_ratio:
            peaks.append((i, float(radial[i])))
    radial_var = float(np.var(radial))
    circularity_score = 1.0 - (radial_var / (radial_var + 1.0))
    return {"radial_profile": radial.tolist(), "rings": peaks, "circularity_score": circularity_score}

# ----------------------------
# Symbol DNA extractor (256)
# ----------------------------
def symbol_dna(arr: np.ndarray, num_angles=128, num_radii=64) -> np.ndarray:
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
    vec = descriptor.mean(axis=0)
    out = np.interp(np.linspace(0, num_radii-1, 256), np.arange(num_radii), vec)
    out = (out - out.min()) / (out.max() - out.min() + 1e-12)
    return out

# ----------------------------
# Audio synth
# ----------------------------
def synth_tone(freq: float, duration: float=1.5, sr:int=22050, harmonics=None, amps=None):
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

# ----------------------------
# Ritual optical + acoustic simulators
# ----------------------------
def ritual_optical_simulator(arr_list, weights, phase_offsets, alpha=2*np.pi):
    fields = []
    for arr in arr_list:
        field = np.exp(1j * alpha * arr)
        fields.append(field)
    combined = np.zeros_like(fields[0], dtype=np.complex128)
    for fld, w, ph in zip(fields, weights, phase_offsets):
        combined += w * fld * np.exp(1j * ph)
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(combined)))
    return safe_image(np.log(1 + intensity.real))

def ritual_acoustic_simulator(spectra, freqs, amplitudes, phase_offsets):
    combined = np.zeros_like(spectra[0], dtype=np.complex128)
    for spec, amp, ph in zip(spectra, amplitudes, phase_offsets):
        combined += amp * spec * np.exp(1j * ph)
    return freqs, np.abs(combined)

# ----------------------------
# Sweep GIF helpers
# ----------------------------
from PIL import Image as PILImage

def build_optical_frame(arrs_for_sweep, weights, phase_offsets_rad, alpha=4*np.pi):
    fields = [np.exp(1j * alpha * a) for a in arrs_for_sweep]
    combined = np.zeros_like(fields[0], dtype=np.complex128)
    for f,w,ph in zip(fields, weights, phase_offsets_rad):
        combined += w * f * np.exp(1j*ph)
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(combined)))
    img = np.log(1 + intensity.real)
    img = img - img.min()
    if img.max()>0:
        img = img / img.max()
    return (img * 255).astype('uint8')

def make_optical_sweep_gif(arrs_for_sweep, weights, start_phase_deg=0, stop_phase_deg=360, frames=24, duration_s=2.0):
    frames_list = []
    for t in range(frames):
        phase = math.radians(start_phase_deg + (stop_phase_deg - start_phase_deg) * (t / float(frames-1)))
        frame = build_optical_frame(arrs_for_sweep, weights, [phase]*len(arrs_for_sweep))
        pil = PILImage.fromarray(frame).convert("L").resize((512,512))
        frames_list.append(pil)
    buf = io.BytesIO()
    frames_list[0].save(buf, format='GIF', save_all=True, append_images=frames_list[1:], duration=int(1000*duration_s/frames), loop=0)
    buf.seek(0)
    return buf.getvalue()

def make_acoustic_sweep_gif(dna_list_sel, freqlow, freqhigh, frames=24, duration_s=2.0):
    freqs = np.linspace(1,2000,2000)
    frames_list = []
    for t in range(frames):
        fbase = freqlow + (freqhigh - freqlow) * (t / float(frames-1))
        combined = np.zeros_like(freqs)
        for dna in dna_list_sel:
            centroid = np.sum(np.arange(len(dna)) * dna) / (dna.sum()+1e-12)
            base = fbase * (0.5 + centroid)
            for mult in [1,2,3]:
                peak = base*mult
                combined += np.exp(-0.5*((freqs - peak)/(peak*0.03+1e-6))**2)
        im = np.zeros((128, 512), dtype=np.uint8)
        env = combined[:800]
        env = env - env.min()
        if env.max()>0:
            env = env / env.max()
        ys = (env * 127).astype(np.uint8)
        for x, y in enumerate(ys):
            im[127-y:, x] = 255
        pil = PILImage.fromarray(im).convert("L")
        frames_list.append(pil)
    buf = io.BytesIO()
    frames_list[0].save(buf, format='GIF', save_all=True, append_images=frames_list[1:], duration=int(1000*duration_s/frames), loop=0)
    buf.seek(0)
    return buf.getvalue()

# ----------------------------
# Export features & DNA ZIP
# ----------------------------
import zipfile
def export_features_and_dna_zip(names, features_list, dna_list, zip_name="indus_export.zip"):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if len(features_list) > 0:
            keys = [k for k in features_list[0].keys() if k != "mag"]
            feats_csv = io.StringIO()
            w = csv.writer(feats_csv)
            w.writerow(["symbol"] + keys)
            for nm, feats in zip(names, features_list):
                row = [nm] + [json.dumps(feats.get(k)) if isinstance(feats.get(k), (dict,list)) else feats.get(k) for k in keys]
                w.writerow(row)
            zf.writestr("features_summary.csv", feats_csv.getvalue())
        for nm, dna in zip(names, dna_list):
            dna_csv = io.StringIO()
            w = csv.writer(dna_csv)
            w.writerow(["idx","dna_value"])
            for i, val in enumerate(dna):
                w.writerow([i, float(val)])
            safe_name = nm.replace(" ", "_")
            zf.writestr(f"{safe_name}_dna.csv", dna_csv.getvalue())
    mem.seek(0)
    return mem.getvalue()

# ----------------------------
# HTML Hypothesis report
# ----------------------------
def generate_html_report(names, features_list, dna_list, cosmic_list):
    parts = ["<html><head><meta charset='utf-8'><title>Indus Hypothesis Report</title></head><body>"]
    parts.append("<h1>Indus Symbol Hypothesis Report</h1>")
    for i, (nm, feats, dna, cosmic) in enumerate(zip(names, features_list, dna_list, cosmic_list)):
        parts.append(f"<h2>{i+1}. {nm}</h2>")
        parts.append("<ul>")
        for k,v in feats.items():
            if k=="mag": continue
            parts.append(f"<li><b>{k}</b>: {json.dumps(v)}</li>")
        parts.append("</ul>")
        # FFT image
        fig, ax = plt.subplots(figsize=(4,2))
        ax.imshow(np.array(feats["mag"]), cmap="magma")
        ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:700px;'>")
        # DNA plot
        fig2, ax2 = plt.subplots(figsize=(6,1.2))
        ax2.plot(dna, linewidth=1)
        ax2.axis("off")
        buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", bbox_inches="tight"); plt.close(fig2); buf2.seek(0)
        b64_2 = base64.b64encode(buf2.read()).decode('ascii')
        parts.append(f"<img src='data:image/png;base64,{b64_2}' style='max-width:700px;'>")
    parts.append("</body></html>")
    return "\n".join(parts)

# ----------------------------
# Session log (in-memory; optional file save)
# ----------------------------
if "session_log" not in st.session_state:
    st.session_state.session_log = []

def record_event(evt):
    st.session_state.session_log.append(evt)

# ----------------------------
# UI: Upload and pipeline
# ----------------------------
st.header("Upload symbols (JPG/PNG) — the pipeline will auto-crop and normalize")
uploads = st.file_uploader("Upload images", accept_multiple_files=True, type=["jpg","jpeg","png"])

# Optionally suggest using the workspace sheet if present (app will not require it)
if os.path.exists(WORKSPACE_SHEET_PATH):
    if st.checkbox("Use workspace sheet (auto-split into 3)"):
        try:
            sheet = Image.open(WORKSPACE_SHEET_PATH).convert("L")
            w,h = sheet.size
            for i in range(3):
                left = int(i * w / 3); right = int((i+1) * w / 3)
                crop = sheet.crop((left,0,right,h))
                buf = io.BytesIO(); crop.save(buf, format='PNG'); buf.seek(0)
                uploads.append(buf)
            st.success("Auto-split sheet loaded into pipeline.")
        except Exception as e:
            st.warning(f"Could not load workspace sheet: {e}")

if not uploads:
    st.info("Upload one or more symbol images to continue.")
    st.stop()

# Produce canonical lists:
arrs = []
names = []
for f in uploads:
    try:
        arr = load_and_normalize(f, target_size=256)
        arrs.append(arr)
        names.append(getattr(f, "name", f"symbol_{len(names)+1}"))
    except Exception as e:
        st.warning(f"Skipping one uploaded file due to error: {e}")

# Feature extraction + dna + cosmic
features_list = []
dna_list = []
cosmic_list = []
for arr in arrs:
    feats = geometry_features_from_image(arr)
    dna = symbol_dna(arr)
    cosmic = detect_circles_and_rings(arr)
    features_list.append(feats)
    dna_list.append(dna)
    cosmic_list.append(cosmic)
    record_event({"type":"analyze","features":feats})

# Display thumbnails and basic inferences
st.header("Preprocessed symbols and quick inferences")
cols = st.columns(len(arrs))
for c, arr, nm, feats, dna, cosmic in zip(cols, arrs, names, features_list, dna_list, cosmic_list):
    c.image(safe_image(arr), caption=nm, use_column_width=True)
    c.write(geometry_code(feats))
    # small inference
    tags = []
    if feats["low_band"] > feats["mid_band"] and feats["low_band"] > feats["high_band"]:
        tags.append("Material/Trade")
    if feats["mid_band"] > feats["low_band"] and feats["mid_band"] > feats["high_band"]:
        tags.append("Civic/Craft")
    if feats["high_band"] > feats["low_band"] and feats["high_band"] > feats["mid_band"]:
        tags.append("Ritual/Abstract")
    if feats["v_score"] > 0.7:
        tags.append("Authority")
    if feats["h_score"] > 0.7:
        tags.append("Cooperation")
    if cosmic["circularity_score"] > 0.6:
        tags.append("Cosmic")
    c.write("Tags: " + ", ".join(tags))

# ----------------------------
# Meaning inference & inspector
# ----------------------------
st.header("Meaning Inference — Inspect & Save mappings")
for i, (nm, feats, dna, cosmic) in enumerate(zip(names, features_list, dna_list, cosmic_list)):
    st.markdown(f"### {i+1}. {nm}")
    st.write("Features:", {k:v for k,v in feats.items() if k!="mag"})
    st.write("Cosmic:", {"circularity_score": cosmic["circularity_score"], "rings": len(cosmic["rings"])})
    st.line_chart(dna)
    # Suggest labels
    fallback = []
    if feats["low_band"] > feats["mid_band"] and feats["low_band"] > feats["high_band"]:
        fallback.append("Material / Trade")
    if feats["mid_band"] > feats["low_band"] and feats["mid_band"] > feats["high_band"]:
        fallback.append("Civic / Craft")
    if feats["high_band"] > feats["low_band"] and feats["high_band"] > feats["mid_band"]:
        fallback.append("Ritual / Abstract")
    st.write("Suggested:", ", ".join(fallback) if fallback else "—")
    label = st.text_input(f"Manual label for {nm}", key=f"label_{i}")
    if st.button(f"Save mapping {nm}", key=f"save_{i}"):
        st.session_state.session_log.append({"type":"mapping","name":nm,"label":label,"features":feats})
        st.success(f"Saved mapping {nm} -> {label}")

# ----------------------------
# Composite / Sequence / Ritual simulator
# ----------------------------
st.header("Composite, Sequence & Ritual simulators")

if len(arrs) >= 2:
    st.subheader("Harmonic Composite (optical sum)")
    w1 = st.slider("Weight A", 0.0, 3.0, 1.0)
    w2 = st.slider("Weight B", 0.0, 3.0, 1.0)
    phase_px = st.slider("Phase shift (px)", 0, 128, 0)
    composite = safe_image(w1 * arrs[0] + w2 * np.roll(arrs[1], int(phase_px), axis=1))
    st.image(composite, caption="Composite", use_column_width=True)

if len(arrs) >= 3:
    st.subheader("Sequence Engine")
    phase_per = st.slider("Phase per symbol (px)", 0, 32, 2)
    out = np.zeros_like(arrs[0])
    for i, a in enumerate(arrs):
        out += np.roll(a, int(i*phase_per), axis=0)
    st.image(safe_image(out), caption="Sequence output", use_column_width=True)

st.subheader("Ritual Activation (optical & acoustic)")
picks = st.multiselect("Pick indices to include", options=list(range(len(arrs))), default=list(range(min(3,len(arrs)))))
if len(picks) > 0:
    weights = [st.slider(f"W for {names[i]}", 0.0, 3.0, 1.0, key=f"w{ i }") for i in picks]
    phases_deg = [st.slider(f"Phase° for {names[i]}", 0, 360, (i*90)%360, key=f"ph{ i }") for i in picks]
    phases_rad = [math.radians(p) for p in phases_deg]
    opt = ritual_optical_simulator([arrs[i] for i in picks], weights, phases_rad, alpha=4*np.pi)
    st.image(opt, caption="Ritual optical interference (FFT intensity)", use_column_width=True)
    # acoustic combine
    freqs = np.linspace(1,1000,1000)
    spectra = []
    for i in picks:
        base = 100 + features_list[i]["lobe_count"]*30 + int(features_list[i]["centroid_shift"]*100)
        spec = np.zeros_like(freqs)
        for mult in [1,2,3]:
            peak = base * mult
            spec += np.exp(-0.5*((freqs - peak)/(peak*0.05+1e-6))**2)
        spectra.append(spec)
    freqs_out, combined_spec = ritual_acoustic_simulator(spectra, freqs, [1.0]*len(spectra), phases_rad)
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(freqs_out, combined_spec)
    ax.set_xlim(0, 600)
    ax.set_xlabel("Hz")
    st.pyplot(fig)
    peak_idx = np.argmax(combined_spec); peak_freq = freqs_out[peak_idx]
    st.write("Dominant combined peak:", int(peak_freq), "Hz")
    if st.button("Play dominant combined tone"):
        st.audio(synth_tone(float(peak_freq), duration=2.0, harmonics=[1,2,3], amps=[1.0,0.4,0.15]), format="audio/wav")

# ----------------------------
# Audio from DNA
# ----------------------------
st.header("Audio: play DNA-derived tone")
idx_choice = st.selectbox("Pick symbol", options=list(range(len(arrs))), format_func=lambda i: names[i])
if idx_choice is not None:
    dna = dna_list[idx_choice]
    centroid = np.sum(np.arange(len(dna)) * dna) / (dna.sum() + 1e-12)
    base_freq = 40 + centroid * 760
    n_harm = st.slider("harmonics", 1, 8, 4)
    harm_mults = [i+1 for i in range(n_harm)]
    harm_amps = [1.0/(i+1) for i in range(n_harm)]
    st.write(f"Suggested base freq: {int(base_freq)} Hz")
    if st.button("Play DNA tone"):
        st.audio(synth_tone(float(base_freq), duration=2.0, harmonics=harm_mults, amps=harm_amps), format="audio/wav")

# ----------------------------
# Sweep gifs & exports
# ----------------------------
st.header("Sweep & Export")

# optical sweep
if len(arrs) >= 1:
    st.subheader("Optical sweep GIF")
    start_deg = st.number_input("start deg", 0, 360, 0)
    end_deg = st.number_input("end deg", 0, 360, 360)
    frames = st.slider("frames", 8, 48, 24)
    duration = st.slider("gif duration (s)", 1.0, 6.0, 2.0)
    pick_indices = st.multiselect("pick indices for sweep", options=list(range(len(arrs))), default=list(range(min(2,len(arrs)))))
    if st.button("Generate optical sweep GIF"):
        arrs_sel = [arrs[i] for i in pick_indices]
        weights = [1.0]*len(arrs_sel)
        gif = make_optical_sweep_gif(arrs_sel, weights, start_deg, end_deg, frames=frames, duration_s=duration)
        st.image(gif)
        st.download_button("Download optical sweep GIF", gif, file_name="optical_sweep.gif", mime="image/gif")

# acoustic sweep
if st.button("Generate acoustic sweep GIF (40→400 Hz)"):
    if len(dna_list) == 0:
        st.info("No DNA available.")
    else:
        gif = make_acoustic_sweep_gif(dna_list[:min(4,len(dna_list))], freqlow=40, freqhigh=400, frames=30, duration_s=3.0)
        st.image(gif)
        st.download_button("Download acoustic sweep GIF", gif, file_name="acoustic_sweep.gif", mime="image/gif")

# Export features + DNA zip
if st.button("Export features & DNA as ZIP"):
    zipb = export_features_and_dna_zip(names, features_list, dna_list)
    st.download_button("Download export ZIP", zipb, file_name="indus_export.zip", mime="application/zip")

# HTML report
if st.button("Generate HTML hypothesis report"):
    html_report = generate_html_report(names, features_list, dna_list, cosmic_list)
    st.download_button("Download HTML report", html_report.encode('utf-8'), file_name="indus_report.html", mime="text/html")

# Session log
st.header("Session log (recent)")
st.write(st.session_state.session_log[-20:])
if st.button("Download session log"):
    st.download_button("Download JSON log", json.dumps(st.session_state.session_log, indent=2).encode('utf-8'), file_name="session_log.json", mime="application/json")

st.success("App ready. Upload symbols and experiment — everything runs on the phone and saves locally where possible.")
