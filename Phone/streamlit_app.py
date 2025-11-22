# app.py
# Indus Resonance Lab — Full unified app with Upload-only Auto-Crop loader
# - Upload-only auto-crop (no mandatory external files)
# - Multi-layer decoding (visible)
# - Auto-learn (on mapping/save/upload) — toggleable
# - Harmonic sequencer, ritual simulator, sweep engines
# - Exports (ZIP, HTML), oscilloscope snapshots, grammar engine
#
# NOTE: A developer-provided workspace path is available as an OPTIONAL checkbox only.
WORKSPACE_SHEET_PATH = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"

import streamlit as st
import numpy as np
from PIL import Image
import io, os, json, csv, math, base64, zipfile, tempfile
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indus Resonance Lab — Unified (Upload-only loader)", layout="wide")
st.title("Indus Symbol Resonance Lab — Unified")

# ------------------------
# Persistence paths
# ------------------------
PERSIST_DIR = "indus_state"
os.makedirs(PERSIST_DIR, exist_ok=True)
RULES_FILE = os.path.join(PERSIST_DIR, "rule_base.json")
MAPPINGS_FILE = os.path.join(PERSIST_DIR, "user_mappings.json")
SESSION_FILE = os.path.join(PERSIST_DIR, "session_log.json")
CENTROIDS_FILE = os.path.join(PERSIST_DIR, "kmeans_centroids.json")

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path,"r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, obj):
    with open(path,"w") as f:
        json.dump(obj, f, indent=2)

# Default rule base (visible multilayer)
DEFAULT_RULES = [
    {"id":"A_container","label":"LayerA:Container/Storage","layer":"A","conditions":{"v_score_min":0.6,"lobe_max":2},"confidence":0.85,"explanation":"vertical symmetry + few lobes → container"},
    {"id":"B_flow","label":"LayerB:Flow/Movement","layer":"B","conditions":{"h_score_min":0.45,"centroid_shift_min":0.03},"confidence":0.78,"explanation":"horizontal symmetry + centroid shift → flow"},
    {"id":"C_deity","label":"LayerC:Ritual/High-harmonic","layer":"C","conditions":{"rot_harmonics_min":4},"confidence":0.92,"explanation":"rotational harmonics → ritual"},
    {"id":"D_operator","label":"LayerD:Operator/Modifier","layer":"D","conditions":{"roughness_min":1e-5},"confidence":0.8,"explanation":"complex roughness → operator"},
    {"id":"E_interf","label":"LayerE:Interference/Gate","layer":"E","conditions":{"lobe_min":3,"rot_harmonics_min":2},"confidence":0.88,"explanation":"multi-lobe + harmonics → gate"},
    {"id":"F_numeric","label":"LayerF:Numeric/Count","layer":"F","conditions":{"lobe_max":1,"centroid_shift_max":0.02},"confidence":0.65,"explanation":"simple centered mark → numeric"}
]

# Session state defaults
if "rule_base" not in st.session_state:
    st.session_state.rule_base = load_json(RULES_FILE, DEFAULT_RULES)
if "user_mappings" not in st.session_state:
    st.session_state.user_mappings = load_json(MAPPINGS_FILE, {})
if "session_log" not in st.session_state:
    st.session_state.session_log = load_json(SESSION_FILE, [])
if "kmeans_centroids" not in st.session_state:
    st.session_state.kmeans_centroids = load_json(CENTROIDS_FILE, {})

if "auto_learn" not in st.session_state:
    st.session_state.auto_learn = True

# ------------------------
# Upload-only auto-crop loader (no external file refs required)
# ------------------------
def auto_multi_crop_from_pil(pil_image, threshold=200, min_area=40):
    """Return list of PIL crops from a PIL image, cropping dark connected components."""
    pil = pil_image.convert("L")
    arr = np.array(pil)
    H, W = arr.shape
    mask = arr < threshold
    visited = np.zeros_like(mask, dtype=bool)
    crops = []
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not visited[y, x]:
                stack = [(y, x)]
                coords = []
                while stack:
                    yy, xx = stack.pop()
                    if yy < 0 or yy >= H or xx < 0 or xx >= W:
                        continue
                    if visited[yy, xx] or not mask[yy, xx]:
                        continue
                    visited[yy, xx] = True
                    coords.append((yy, xx))
                    stack.extend([(yy+1,xx),(yy-1,xx),(yy,xx+1),(yy,xx-1)])
                if len(coords) >= min_area:
                    ys = [c[0] for c in coords]; xs = [c[1] for c in coords]
                    y0, y1 = max(min(ys)-2, 0), min(max(ys)+2, H)
                    x0, x1 = max(min(xs)-2, 0), min(max(xs)+2, W)
                    crop = pil.crop((x0, y0, x1, y1))
                    crops.append(crop)
    return crops

def load_and_normalize_pil(pil_image, size=256, autotrim=True):
    if autotrim:
        pil_image = pil_image.convert("L")
        pil_image = pil_image.crop(pil_image.getbbox()) if pil_image.getbbox() else pil_image
    img = pil_image.convert("L")
    w, h = img.size
    scale = size / max(w, h)
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    canvas = Image.new("L", (size, size), 255)
    offset = ((size - img.size[0])//2, (size - img.size[1])//2)
    canvas.paste(img, offset)
    arr = np.array(canvas).astype(np.float32)
    if arr.max() > 0:
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
    return arr

def safe_image(arr):
    a = np.nan_to_num(arr)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return a

# ------------------------
# Geometry & DNA & detectors
# ------------------------
def geometry_features_from_image(arr):
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
        "mag": mag.tolist()
    }

def symbol_dna(arr, num_angles=128, num_radii=64):
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

def detect_circles_and_rings(arr, threshold_ratio=0.25):
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

# ------------------------
# Rule engine & learner (k-NN)
# ------------------------
def build_knn(mappings):
    X = []; y = []
    for nm, rec in mappings.items():
        f = rec["features"]
        vec = np.array([min(f["lobe_count"],8)/8.0, f["v_score"], f["h_score"], min(f["rot_harmonics"],8)/8.0, f["centroid_shift"]])
        X.append(vec); y.append(rec["saved_label"])
    if len(X)==0:
        return np.zeros((0,5)), []
    return np.vstack(X), y

def knn_predict_one(x, X_train, y_train, k=3):
    if X_train.shape[0]==0:
        return None, 0.0
    d = np.linalg.norm(X_train - x.reshape(1,-1), axis=1)
    idx = np.argsort(d)[:k]
    labs = [y_train[i] for i in idx]
    from collections import Counter
    cnt = Counter(labs)
    lab, count = cnt.most_common(1)[0]
    return lab, count/float(k)

def rule_matches(features, rule):
    c = rule.get("conditions", {})
    detail = {}
    if "v_score_min" in c and features["v_score"] < c["v_score_min"]:
        detail["v_score"]=(features["v_score"], c["v_score_min"]); return False, detail
    if "v_score_max" in c and features["v_score"] > c["v_score_max"]:
        detail["v_score"]=(features["v_score"], c["v_score_max"]); return False, detail
    if "h_score_min" in c and features["h_score"] < c["h_score_min"]:
        detail["h_score"]=(features["h_score"], c["h_score_min"]); return False, detail
    if "h_score_max" in c and features["h_score"] > c["h_score_max"]:
        detail["h_score"]=(features["h_score"], c["h_score_max"]); return False, detail
    if "lobe_min" in c and features["lobe_count"] < c["lobe_min"]:
        detail["lobe_count"]=(features["lobe_count"], c["lobe_min"]); return False, detail
    if "lobe_max" in c and features["lobe_count"] > c["lobe_max"]:
        detail["lobe_count"]=(features["lobe_count"], c["lobe_max"]); return False, detail
    if "rot_harmonics_min" in c and features["rot_harmonics"] < c["rot_harmonics_min"]:
        detail["rot_harmonics"]=(features["rot_harmonics"], c["rot_harmonics_min"]); return False, detail
    if "roughness_min" in c and features["roughness"] < c["roughness_min"]:
        detail["roughness"]=(features["roughness"], c["roughness_min"]); return False, detail
    if "centroid_shift_min" in c and features["centroid_shift"] < c["centroid_shift_min"]:
        detail["centroid_shift"]=(features["centroid_shift"], c["centroid_shift_min"]); return False, detail
    if "centroid_shift_max" in c and features["centroid_shift"] > c["centroid_shift_max"]:
        detail["centroid_shift"]=(features["centroid_shift"], c["centroid_shift_max"]); return False, detail
    return True, {}

PROTOTYPES = {
    "Container/Storage": np.array([1.0,0.8,0.2,0.5,0.02]),
    "Flow/Movement": np.array([0.3,0.2,0.8,0.7,0.12]),
    "Duality/Pairing": np.array([0.7,0.4,0.4,0.9,0.04]),
    "Cosmic/Celestial": np.array([1.0,0.5,0.5,1.0,0.01]),
    "Numeric/Count": np.array([0.05,0.95,0.95,0.05,0.005])
}

def fallback_predict(features):
    fv = np.array([min(features["lobe_count"],8)/8.0, features["v_score"], features["h_score"], min(features["rot_harmonics"],8)/8.0, features["centroid_shift"]])
    dists = {}
    for label, proto in PROTOTYPES.items():
        d = np.linalg.norm(fv - proto)
        dists[label] = float(d)
    maxd = max(dists.values()) if dists else 1.0
    sims = {k: 1 - (v/(maxd+1e-12)) for k,v in dists.items()}
    return sorted(sims.items(), key=lambda x:-x[1])

def multilayer_read(arr, name, pipeline_mode="hybrid"):
    feats = geometry_features_from_image(arr)
    dna = symbol_dna(arr)
    cosmic = detect_circles_and_rings(arr)
    rule_hits = []
    for r in st.session_state.rule_base:
        ok, detail = rule_matches(feats, r)
        if ok:
            rule_hits.append({"label": r["label"], "confidence": r["confidence"], "explanation": r["explanation"], "rule_id": r["id"]})
    fallback = fallback_predict(feats)
    X_train, y_train = build_knn(st.session_state.user_mappings)
    xq = np.array([min(feats["lobe_count"],8)/8.0, feats["v_score"], feats["h_score"], min(feats["rot_harmonics"],8)/8.0, feats["centroid_shift"]])
    knn_label, knn_conf = knn_predict_one(xq, X_train, y_train, k=min(3, max(1, X_train.shape[0]))) if X_train.shape[0]>0 else (None, 0.0)
    dna_scores = []
    for nm, rec in st.session_state.user_mappings.items():
        if "dna" in rec:
            proto = np.array(rec["dna"])
            score = np.dot(proto, dna) / ((np.linalg.norm(proto)+1e-12)*(np.linalg.norm(dna)+1e-12))
            dna_scores.append((rec["saved_label"], float(score)))
    dna_scores_sorted = sorted(dna_scores, key=lambda x:-x[1])
    combined = {}
    for lab, sim in fallback:
        combined[lab] = combined.get(lab, 0.0) + 0.25*sim
    if pipeline_mode in ("preload","hybrid"):
        for rh in rule_hits:
            combined[rh["label"]] = combined.get(rh["label"], 0.0) + 0.6*rh["confidence"]
    if pipeline_mode in ("learn","hybrid") and knn_label:
        combined[knn_label] = combined.get(knn_label, 0.0) + 0.6*knn_conf
    for lab, sc in dna_scores_sorted[:3]:
        combined[lab] = combined.get(lab, 0.0) + 0.35*sc
    ranked = sorted(combined.items(), key=lambda x:-x[1])
    explanation = {"features": feats, "rule_hits": rule_hits, "fallback": fallback[:5], "knn": (knn_label, knn_conf), "dna_scores": dna_scores_sorted[:5], "cosmic": cosmic}
    return ranked, explanation, dna

# ------------------------
# Utility: optical & acoustic sweep frames
# ------------------------
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
        pil = Image.fromarray(frame).convert("L").resize((512,512))
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
        im = np.zeros((128,512), dtype=np.uint8)
        env = combined[:800]
        env = env - env.min()
        if env.max()>0:
            env = env / env.max()
        ys = (env * 127).astype(np.uint8)
        for x, y in enumerate(ys):
            im[127-y:, x] = 255
        pil = Image.fromarray(im).convert("L")
        frames_list.append(pil)
    buf = io.BytesIO()
    frames_list[0].save(buf, format='GIF', save_all=True, append_images=frames_list[1:], duration=int(1000*duration_s/frames), loop=0)
    buf.seek(0)
    return buf.getvalue()

# ------------------------
# Energy flow transform
# ------------------------
def energy_flow_transform(arrA, arrB, alpha=1.0):
    FA = np.fft.fft2(arrA)
    FB = np.fft.fft2(arrB)
    magA = np.abs(FA)
    phA = np.angle(FA)
    maskB = 1.0 / (1.0 + np.exp(-10*(arrB - 0.5)))
    low_B = np.abs(np.fft.ifft2(np.fft.fft2(maskB) * (np.exp(-(np.fft.fftfreq(arrB.shape[0])**2)[:,None]*10))))
    transformed_mag = magA * (1.0 + alpha * low_B)
    transformed = np.real(np.fft.ifft2(transformed_mag * np.exp(1j*phA)))
    return safe_image(transformed)

# ------------------------
# Exports / Reports
# ------------------------
def export_features_and_dna_zip(names, features_list, dna_list):
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if len(features_list)>0:
            keys = [k for k in features_list[0].keys() if k!="mag"]
            s = io.StringIO(); w = csv.writer(s)
            w.writerow(["symbol"] + keys)
            for nm, feats in zip(names, features_list):
                row = [nm] + [json.dumps(feats.get(k)) if isinstance(feats.get(k),(list,dict)) else feats.get(k) for k in keys]
                w.writerow(row)
            zf.writestr("features_summary.csv", s.getvalue())
        for nm, dna in zip(names, dna_list):
            s = io.StringIO(); w = csv.writer(s)
            w.writerow(["idx","dna_value"])
            for i,v in enumerate(dna):
                w.writerow([i, float(v)])
            safe_name = nm.replace(" ","_")
            zf.writestr(f"{safe_name}_dna.csv", s.getvalue())
    mem.seek(0)
    return mem.getvalue()

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
        fig, ax = plt.subplots(figsize=(4,2)); ax.imshow(np.array(feats["mag"]), cmap="magma"); ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:700px;'>")
        fig2, ax2 = plt.subplots(figsize=(6,1.2)); ax2.plot(dna); ax2.axis("off")
        buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", bbox_inches="tight"); plt.close(fig2); buf2.seek(0)
        b64_2 = base64.b64encode(buf2.read()).decode('ascii')
        parts.append(f"<img src='data:image/png;base64,{b64_2}' style='max-width:700px;'>")
    parts.append("</body></html>")
    return "\n".join(parts)

# ------------------------
# Main UI: Upload, process, read, simulate, export
# ------------------------
st.sidebar.header("Engine & UX")
st.sidebar.write("Tip: Upload symbol images (or sheets). The app auto-detects dark glyphs and crops them.")
st.sidebar.checkbox("Include developer workspace file (optional)", value=False, key="include_workspace")  # off by default
st.sidebar.checkbox("Auto-learn on mapping/save/upload", value=st.session_state.auto_learn, key="auto_learn")
pipeline_choice = st.sidebar.selectbox("Pipeline mode", ["hybrid (rules + learn)", "preload (rules only)", "learn (k-NN only)"])

# Upload component (uploads only)
st.header("Upload & Auto-split (uploads only)")
uploaded_files = st.file_uploader("Upload one or more JPG/PNG images", type=["jpg","jpeg","png"], accept_multiple_files=True)

# If user opted in explicitly, optionally add the developer-provided workspace file into the processing queue.
# It is *optional and disabled by default*. We include it only when the user checks the checkbox.
if st.session_state.get("include_workspace", False):
    # Ensure workspace exists before adding
    if os.path.exists(WORKSPACE_SHEET_PATH):
        try:
            with open(WORKSPACE_SHEET_PATH, "rb") as f:
                b = io.BytesIO(f.read()); b.name = os.path.basename(WORKSPACE_SHEET_PATH)
                if uploaded_files:
                    uploaded_files = list(uploaded_files)  # make mutable
                    uploaded_files.append(b)
                else:
                    uploaded_files = [b]
            st.info("Workspace file added to uploads (you enabled this).")
        except Exception as e:
            st.warning(f"Workspace file could not be read: {e}")
    else:
        st.warning("Workspace file not found in environment; nothing added.")

if not uploaded_files:
    st.info("Upload at least one image to continue (auto-crop will run on the uploaded images).")
    st.stop()

# Option: auto multicrop checkbox and parameters
do_auto_multi = st.checkbox("Auto-detect and split glyphs inside uploaded images", value=True)
threshold = st.slider("Crop threshold (lower detects lighter marks)", 50, 255, 200)
min_area = st.number_input("Min component area (pixels)", min_value=10, max_value=5000, value=40)

# Build processing list from uploads
processing = []
for f in uploaded_files:
    raw = f.read() if hasattr(f, "read") else f
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        pil = Image.open(io.BytesIO(raw))
    if do_auto_multi:
        comps = auto_multi_crop_from_pil(pil.convert("L"), threshold=threshold, min_area=min_area)
        if comps:
            processing.extend(comps)
        else:
            # If no comps found, keep the original upload as one item
            processing.append(pil)
    else:
        processing.append(pil)

# Normalize processing items into canonical arrays and names
arrs = []; names = []
for i, item in enumerate(processing):
    try:
        pil = item if isinstance(item, Image.Image) else Image.open(item)
        arr = load_and_normalize_pil(pil, size=256, autotrim=True)
        arrs.append(arr)
        names.append(getattr(item, "name", f"symbol_{len(names)+1}"))
    except Exception as e:
        st.warning(f"Skipping one item due to load error: {e}")

# compute features, dna, cosmic
features_list = []; dna_list = []; cosmic_list = []
for arr in arrs:
    feats = geometry_features_from_image(arr)
    dna = symbol_dna(arr)
    cosmic = detect_circles_and_rings(arr)
    features_list.append(feats); dna_list.append(dna); cosmic_list.append(cosmic)
    st.session_state.session_log.append({"type":"analyze","features":feats})

# thumbnails
st.header("Detected & normalized symbols")
cols = st.columns(min(6, len(arrs) if arrs else 1))
for c, a, n in zip(cols, arrs, names):
    c.image(safe_image(a), caption=n, use_column_width=True)

# show the multilayer rule base (visible)
st.header("Multi-layer decoding framework (visible)")
with st.expander("Show/Modify preloaded rule base"):
    for i, r in enumerate(st.session_state.rule_base):
        st.markdown(f"**{r['label']}** (layer {r.get('layer','?')}) — confidence {r.get('confidence',0.6):.2f}")
        st.write("Conditions:", r.get("conditions", {}))
        st.write("Explanation:", r.get("explanation"))
    if st.button("Reset rules to defaults"):
        st.session_state.rule_base = DEFAULT_RULES.copy()
        save_json(RULES_FILE, st.session_state.rule_base)
        st.success("Rules reset.")

# Run reads for each symbol and allow save-label (human-in-loop)
st.header("Reads, inference & human-in-the-loop mapping")
mode_map = {"hybrid (rules + learn)":"hybrid","preload (rules only)":"preload","learn (k-NN only)":"learn"}
pipeline_mode = mode_map[pipeline_choice]

for i, (arr, nm) in enumerate(zip(arrs, names)):
    ranked, explanation, dna = multilayer_read(arr, nm, pipeline_mode)
    st.subheader(f"{i+1}. {nm}")
    if ranked:
        for lab, sc in ranked[:6]:
            st.write(f"- {lab} — score {sc:.2f}")
    else:
        st.write("_No hypotheses_")
    st.write("Explanation:")
    st.json(explanation)
    label_field = st.text_input(f"Manual label for {nm} (optional)", key=f"label_{i}")
    if st.button(f"Save mapping for {nm}", key=f"save_{i}"):
        if label_field.strip():
            st.session_state.user_mappings[nm] = {"saved_label": label_field.strip(), "features": explanation["features"], "dna": dna.tolist()}
            save_json(MAPPINGS_FILE, st.session_state.user_mappings)
            st.success(f"Saved mapping: {nm} → {label_field.strip()}")
            st.session_state.session_log.append({"type":"mapping","symbol":nm,"label":label_field.strip()})
            # auto-learn: ensure dna stored
            if st.session_state.get("auto_learn", True):
                if "dna" not in st.session_state.user_mappings[nm]:
                    st.session_state.user_mappings[nm]["dna"] = dna.tolist()
                    save_json(MAPPINGS_FILE, st.session_state.user_mappings)
                    st.success("Auto-learn: DNA saved to mappings.")
        else:
            st.info("Enter a non-empty label to save mapping.")

# If auto-learn is enabled, ensure any mappings missing dna get dna from current batch if names match
if st.session_state.get("auto_learn", True):
    for name, rec in st.session_state.user_mappings.items():
        if "dna" not in rec:
            if name in names:
                idx = names.index(name)
                st.session_state.user_mappings[name]["dna"] = dna_list[idx].tolist()
                save_json(MAPPINGS_FILE, st.session_state.user_mappings)

# Multi-symbol harmonic sequencer
st.header("Multi-Symbol Harmonic Sequencer")
sel_idxs = st.multiselect("Choose symbol indices", options=list(range(len(arrs))), default=list(range(min(3,len(arrs)))))
if len(sel_idxs) > 0:
    base_freqs = {idx: st.number_input(f"Base freq for {names[idx]}", min_value=20, max_value=2000, value=100 + features_list[idx]["lobe_count"]*30, key=f"bf_{idx}") for idx in sel_idxs}
    weights = [st.slider(f"Weight {names[idx]}", 0.0, 3.0, 1.0, key=f"w_{idx}") for idx in sel_idxs]
    phase_deg = [st.slider(f"Phase° {names[idx]}", 0, 360, 0, key=f"ph_{idx}") for idx in sel_idxs]
    if st.button("Run harmonic sequencer"):
        freqs = np.linspace(1,2000,2000)
        combined = np.zeros_like(freqs)
        for idx in sel_idxs:
            base = base_freqs[idx]; w = weights[sel_idxs.index(idx)]
            for mult in [1,2,3]:
                peak = base * mult
                combined += w * (1.0/mult) * np.exp(-0.5*((freqs - peak)/(peak*0.03+1e-6))**2)
        fig, ax = plt.subplots(figsize=(6,2)); ax.plot(freqs, combined); ax.set_xlim(0,800); st.pyplot(fig)
        comp = np.zeros_like(arrs[0])
        for idx in sel_idxs:
            comp += weights[sel_idxs.index(idx)] * np.roll(arrs[idx], int(phase_deg[sel_idxs.index(idx)]/360*arrs[idx].shape[1]), axis=1)
        st.image(safe_image(comp), caption="Optical composite")

# AC/DC sweep engine
st.header("AC/DC Frequency Sweep Engine (acoustic GIF)")
sweep_mode = st.selectbox("Sweep mode", ["linear","log","chirp","burst"])
sweep_from = st.number_input("Start Hz", 1, 1, 30)
sweep_to = st.number_input("Stop Hz", 1, 2000, 400)
sweep_frames = st.slider("Frames", 8, 60, 24)
sweep_duration = st.slider("GIF duration (s)", 1.0, 6.0, 2.0)
if st.button("Generate acoustic sweep GIF"):
    gif = make_acoustic_sweep_gif(dna_list[:min(4,len(dna_list))], freqlow=sweep_from, freqhigh=sweep_to, frames=sweep_frames, duration_s=sweep_duration)
    st.image(gif)
    st.download_button("Download acoustic sweep GIF", gif, file_name="acoustic_sweep.gif", mime="image/gif")

# Symbol→Symbol energy flow
st.header("Symbol → Symbol Energy Flow Simulator")
if len(arrs) >= 2:
    aidx = st.selectbox("From symbol", options=list(range(len(arrs))), format_func=lambda i: names[i])
    bidx = st.selectbox("To symbol", options=list(range(len(arrs))), format_func=lambda i: names[i], index=(1 if len(arrs)>1 else 0))
    alpha = st.slider("Transform strength", 0.0, 3.0, 1.0)
    if st.button("Simulate A→B"):
        tarr = energy_flow_transform(arrs[aidx], arrs[bidx], alpha=alpha)
        col1, col2, col3 = st.columns(3)
        col1.image(safe_image(arrs[aidx]), caption=f"A: {names[aidx]}")
        col2.image(safe_image(arrs[bidx]), caption=f"B: {names[bidx]}")
        col3.image(tarr, caption="A→B transformed")
        st.session_state.session_log.append({"type":"energy_flow","from":names[aidx],"to":names[bidx],"alpha":alpha})

# Exports & report
st.header("Exports & HTML report")
if st.button("Export features & DNA ZIP"):
    zipb = export_features_and_dna_zip(names, features_list, dna_list)
    st.download_button("Download ZIP", zipb, file_name="indus_export.zip", mime="application/zip")
if st.button("Generate HTML Hypothesis Report"):
    html_rep = generate_html_report(names, features_list, dna_list, cosmic_list)
    st.download_button("Download HTML report", html_rep.encode('utf-8'), file_name="indus_hypothesis_report.html", mime="text/html")
    st.success("Report generated. Open it in a browser and Save As PDF if you need a PDF.")

# Oscilloscope snapshot
st.header("Oscilloscope & Spectrogram (snapshot)")
if len(arrs) > 0:
    osc_idx = st.selectbox("Select symbol for snapshot", options=list(range(len(arrs))), format_func=lambda i: names[i])
    if st.button("Generate osc snapshot"):
        dna = dna_list[osc_idx]
        centroid = np.sum(np.arange(len(dna)) * dna) / (dna.sum() + 1e-12)
        base = 40 + centroid * 760
        sr = 22050; dur = 1.0
        t = np.linspace(0, dur, int(sr*dur), endpoint=False)
        sig = np.zeros_like(t)
        for m in range(1,5):
            sig += (1.0/m) * np.sin(2*np.pi*base*m*t)
        fig, axs = plt.subplots(2,1, figsize=(8,3))
        axs[0].plot(t[:1000], sig[:1000]); axs[0].set_title("Oscilloscope")
        axs[1].specgram(sig, NFFT=1024, Fs=sr, noverlap=512); axs[1].set_title("Spectrogram")
        st.pyplot(fig)
        st.session_state.session_log.append({"type":"osc_snapshot","symbol":names[osc_idx],"base":base})

# Grammar engine (experimental)
st.header("Indus Grammar Engine (experimental)")
seq_input = st.text_input("Enter sequence indices (e.g. 0,1,2)", value=",".join(str(i) for i in range(min(3,len(arrs)))))
if st.button("Analyze sequence grammar"):
    try:
        seq = [int(x.strip()) for x in seq_input.split(",") if x.strip()!='']
        roles = []
        for idx in seq:
            f = features_list[idx]
            role = "unknown"
            if f["rot_harmonics"] >= 4:
                role = "operator"
            elif f["v_score"] > 0.7:
                role = "carrier"
            elif f["centroid_shift"] > 0.12:
                role = "modifier"
            elif f["lobe_count"] <= 1:
                role = "numeral"
            roles.append((names[idx], role))
        st.write("Roles:")
        for r in roles:
            st.write(f"- {r[0]} → {r[1]}")
        st.session_state.session_log.append({"type":"grammar","seq":seq,"roles":[r[1] for r in roles]})
    except Exception as e:
        st.error(f"Error parsing sequence: {e}")

# Session log controls
st.header("Session log & persistence")
st.write("Recent events:")
st.write(st.session_state.session_log[-20:])
if st.button("Save session to disk"):
    save_json(SESSION_FILE, st.session_state.session_log); st.success(f"Saved session to {SESSION_FILE}")
if st.button("Save mappings to disk"):
    save_json(MAPPINGS_FILE, st.session_state.user_mappings); st.success(f"Saved mappings to {MAPPINGS_FILE}")
if st.button("Clear session"):
    st.session_state.session_log = []; save_json(SESSION_FILE, st.session_state.session_log); st.success("Cleared session")

st.success("App ready. Upload images and the loader will auto-crop glyphs from uploads only. The developer sheet path is optional and disabled by default.")
