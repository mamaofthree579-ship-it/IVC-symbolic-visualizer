# app.py
# Indus Resonance Lab — Full Upgrades (multi-layer engine + all requested features)
# Paste into app.py and run: streamlit run app.py
# Optional workspace sheet path (only used if you enable it in UI)
WORKSPACE_SHEET_PATH = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io, os, json, csv, math, base64, tempfile, time
import matplotlib.pyplot as plt

st.set_page_config(page_title="Indus Resonance Lab — Ultimate", layout="wide")
st.title("Indus Symbol Resonance Lab — Ultimate (All Upgrades)")

# --------------------
# Utilities: file / persist
# --------------------
PERSIST_DIR = "indus_state"
os.makedirs(PERSIST_DIR, exist_ok=True)
RULES_FILE = os.path.join(PERSIST_DIR, "rule_base.json")
MAPPINGS_FILE = os.path.join(PERSIST_DIR, "user_mappings.json")
SESSION_FILE = os.path.join(PERSIST_DIR, "session_log.json")
CENTROIDS_FILE = os.path.join(PERSIST_DIR, "kmeans_centroids.json")

def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        return default
    return default

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# session_persistent storage
if "rule_base" not in st.session_state:
    st.session_state.rule_base = load_json(RULES_FILE, [
        {"id":"rule_container","label":"Container/Storage","conditions":{"v_score_min":0.6,"h_score_max":0.6,"lobe_max":2},"confidence":0.85,"explanation":"vertical symmetry + few lobes"},
        {"id":"rule_flow","label":"Flow/Movement","conditions":{"h_score_min":0.45,"centroid_shift_min":0.03},"confidence":0.78,"explanation":"horizontal symmetry + centroid shift"},
        {"id":"rule_pair","label":"Duality/Pairing","conditions":{"lobe_min":3,"rot_harmonics_min":2},"confidence":0.88,"explanation":"multiple lobes"},
        {"id":"rule_celestial","label":"Cosmic/Celestial","conditions":{"rot_harmonics_min":4,"lobe_min":4},"confidence":0.92,"explanation":"rich rotational harmonic structure"},
        {"id":"rule_numeric","label":"Numeric/Count","conditions":{"lobe_max":1,"centroid_shift_max":0.02},"confidence":0.6,"explanation":"simple centered mark"}
    ])

if "user_mappings" not in st.session_state:
    st.session_state.user_mappings = load_json(MAPPINGS_FILE, {})

if "session_log" not in st.session_state:
    st.session_state.session_log = load_json(SESSION_FILE, [])

if "kmeans_centroids" not in st.session_state:
    st.session_state.kmeans_centroids = load_json(CENTROIDS_FILE, {})

# --------------------
# Image helpers: autocrop, normalize, load
# --------------------
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

def load_and_normalize(img_file, target_size=256, autocrop_tol=10, autotrim=True):
    # accepts path or streamlit UploadedFile
    if isinstance(img_file, str):
        img = Image.open(img_file).convert("L")
    else:
        img_file.seek(0)
        img = Image.open(img_file).convert("L")
    if autotrim:
        img = autocrop(img, tol=autocrop_tol)
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

def safe_image(arr):
    a = np.nan_to_num(arr)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return a

# --------------------
# Geometry & features
# --------------------
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

def geometry_code(features):
    return f"G{features['lobe_count']}-V{features['v_score']:.2f}-H{features['h_score']:.2f}-R{features['rot_harmonics']}"

# --------------------
# Cosmic detector
# --------------------
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

# --------------------
# Symbol DNA
# --------------------
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

# --------------------
# Simple k-NN (NumPy) classifier using features
# --------------------
def build_knn(mappings):
    X = []
    y = []
    for k,v in mappings.items():
        f = v["features"]
        vec = np.array([min(f["lobe_count"],8)/8.0, f["v_score"], f["h_score"], min(f["rot_harmonics"],8)/8.0, f["centroid_shift"]])
        X.append(vec); y.append(v["saved_label"])
    if len(X)==0:
        return np.zeros((0,5)), []
    return np.vstack(X), y

def knn_predict_one(x, X_train, y_train, k=3):
    if X_train.shape[0] == 0:
        return None, 0.0
    dists = np.linalg.norm(X_train - x.reshape(1,-1), axis=1)
    idx = np.argsort(dists)[:k]
    labels = [y_train[i] for i in idx]
    from collections import Counter
    cnt = Counter(labels)
    label, count = cnt.most_common(1)[0]
    conf = count / float(k)
    return label, conf

# --------------------
# Rule engine helpers
# --------------------
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
    if "centroid_shift_min" in c and features["centroid_shift"] < c["centroid_shift_min"]:
        detail["centroid_shift"]=(features["centroid_shift"], c["centroid_shift_min"]); return False, detail
    if "centroid_shift_max" in c and features["centroid_shift"] > c["centroid_shift_max"]:
        detail["centroid_shift"]=(features["centroid_shift"], c["centroid_shift_max"]); return False, detail
    return True, {}

# --------------------
# Multi-layer reading pipeline (runs all layers & returns ranked hypotheses)
# --------------------
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
    """pipeline_mode: 'preload' (rules only), 'learn' (knn only), 'hybrid' (combine)"""
    feats = geometry_features_from_image(arr)
    dna = symbol_dna(arr)
    cosmic = detect_circles_and_rings(arr)
    results = []
    # 1) Rule-based
    rule_hits = []
    for r in st.session_state.rule_base:
        ok, detail = rule_matches(feats, r)
        if ok:
            rule_hits.append({"label": r["label"], "confidence": r["confidence"], "explanation": r["explanation"], "rule_id": r["id"]})
    # 2) Fallback prototype
    fallback = fallback_predict(feats)
    # 3) k-NN (if mappings exist)
    X_train, y_train = build_knn(st.session_state.user_mappings)
    xq = np.array([min(feats["lobe_count"],8)/8.0, feats["v_score"], feats["h_score"], min(feats["rot_harmonics"],8)/8.0, feats["centroid_shift"]])
    knn_label, knn_conf = knn_predict_one(xq, X_train, y_train, k=min(3, max(1, X_train.shape[0]))) if X_train.shape[0]>0 else (None, 0.0)
    # 4) DNA similarity (cosine) to saved mappings
    dna_scores = []
    for k,v in st.session_state.user_mappings.items():
        if "dna" in v:
            proto = np.array(v["dna"])
            score = np.dot(proto, dna) / (np.linalg.norm(proto)+1e-12) / (np.linalg.norm(dna)+1e-12)
            dna_scores.append((v["saved_label"], float(score)))
    dna_scores_sorted = sorted(dna_scores, key=lambda x:-x[1])
    # Compose results according to pipeline_mode
    combined = {}
    # start with fallback prototypes (weight 0.3)
    for label, sim in fallback:
        combined[label] = combined.get(label, 0.0) + 0.3 * sim
    # rules (weight 0.7)
    if pipeline_mode in ("preload","hybrid"):
        for rh in rule_hits:
            combined[rh["label"]] = combined.get(rh["label"], 0.0) + 0.7 * rh["confidence"]
    # knn (weight 0.6)
    if pipeline_mode in ("learn","hybrid") and knn_label:
        combined[knn_label] = combined.get(knn_label, 0.0) + 0.6 * knn_conf
    # dna matches (weight 0.4)
    for lab, sc in dna_scores_sorted[:3]:
        combined[lab] = combined.get(lab, 0.0) + 0.4 * sc
    # rank
    ranked = sorted(combined.items(), key=lambda x:-x[1])
    explanation = {"features": feats, "rule_hits": rule_hits, "fallback": fallback[:5], "knn": (knn_label, knn_conf), "dna_scores": dna_scores_sorted[:5], "cosmic": cosmic}
    return ranked, explanation, dna

# --------------------
# Sweep generation helpers: optical & acoustic (reused later)
# --------------------
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
        im = np.zeros((128, 512), dtype=np.uint8)
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

# --------------------
# Energy flow simulator: transform A->B via geometry convolution-style model
# --------------------
def energy_flow_transform(arrA, arrB, alpha=1.0):
    # approximate transform: convolve A's FFT magnitude with B's geometry mask, then inverse
    FA = np.fft.fft2(arrA)
    FB = np.fft.fft2(arrB)
    magA = np.abs(FA)
    phA = np.angle(FA)
    # create mask from B (sigmoid of its image)
    maskB = 1.0 / (1.0 + np.exp(-10*(arrB - 0.5)))
    # modulate magA by maskB's lowpass of its fft
    low_B = np.abs(np.fft.ifft2(np.fft.fft2(maskB) * (np.exp(- (np.fft.fftfreq(arrB.shape[0])**2)[:,None]*10))))
    transformed_mag = magA * (1.0 + alpha * low_B)
    transformed = np.real(np.fft.ifft2(transformed_mag * np.exp(1j*phA)))
    return safe_image(transformed)

# --------------------
# Export: features & DNA ZIP + HTML report
# --------------------
def export_features_and_dna_zip(names, features_list, dna_list):
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
        fig, ax = plt.subplots(figsize=(4,2))
        ax.imshow(np.array(feats["mag"]), cmap="magma")
        ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        parts.append(f"<img src='data:image/png;base64,{b64}' style='max-width:700px;'>")
        fig2, ax2 = plt.subplots(figsize=(6,1.2))
        ax2.plot(dna, linewidth=1)
        ax2.axis("off")
        buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", bbox_inches="tight"); plt.close(fig2); buf2.seek(0)
        b64_2 = base64.b64encode(buf2.read()).decode('ascii')
        parts.append(f"<img src='data:image/png;base64,{b64_2}' style='max-width:700px;'>")
    parts.append("</body></html>")
    return "\n".join(parts)

# --------------------
# Main UI: pipeline control and execution
# --------------------
st.sidebar.header("Pipeline Mode")
pipeline_mode = st.sidebar.radio("Choose pipeline mode", ["hybrid (recommended)", "preload (rules only)", "learn (k-NN only)"])

# upload + improved multi-crop
st.header("Upload images (multi-crop auto-detection enabled)")
uploads = st.file_uploader("Upload images (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)

# improved auto-cropper: detect connected components (simple threshold) and crop each as separate symbol
def auto_multi_crop(img_bytes, threshold=250, min_area=50):
    """Return list of bytes objects for each detected component (simple)"""
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    arr = np.array(img)
    # binarize
    mask = arr < threshold
    # label connected components via simple flood fill
    visited = np.zeros(mask.shape, dtype=bool)
    comps = []
    H, W = mask.shape
    for y in range(H):
        for x in range(W):
            if mask[y,x] and not visited[y,x]:
                # flood
                stack = [(y,x)]
                coords = []
                while stack:
                    yy, xx = stack.pop()
                    if yy < 0 or yy>=H or xx<0 or xx>=W: continue
                    if visited[yy,xx] or not mask[yy,xx]: continue
                    visited[yy,xx] = True
                    coords.append((yy,xx))
                    stack.append((yy+1,xx)); stack.append((yy-1,xx)); stack.append((yy,xx+1)); stack.append((yy,xx-1))
                if len(coords) >= min_area:
                    ys = [c[0] for c in coords]; xs = [c[1] for c in coords]
                    y0, y1 = max(min(ys)-2,0), min(max(ys)+2,H)
                    x0, x1 = max(min(xs)-2,0), min(max(xs)+2,W)
                    crop = img.crop((x0,y0,x1,y1))
                    buf = io.BytesIO(); crop.save(buf, format='PNG'); buf.seek(0)
                    comps.append(buf)
    return comps

# assemble processing list with optional auto-multi-crop
processing = []
if uploads:
    auto_crop_checkbox = st.checkbox("Auto-detect multiple glyphs inside uploaded images and split into symbols", value=True)
    for f in uploads:
        raw = f.read()
        if auto_crop_checkbox:
            comps = auto_multi_crop(raw)
            if comps:
                processing.extend(comps)
            else:
                # fallback to original
                processing.append(io.BytesIO(raw))
        else:
            processing.append(io.BytesIO(raw))

if len(processing) == 0:
    st.info("Upload at least one image to process.")
    st.stop()

# preprocess all into canonical arrs, names
arrs = []
names = []
for i, buf in enumerate(processing):
    try:
        arr = load_and_normalize(buf, target_size=256)
        arrs.append(arr)
        names.append(getattr(buf, "name", f"symbol_{i+1}"))
    except Exception as e:
        st.warning(f"Skipping one item (load error): {e}")

# compute features, dna, cosmic
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

# display thumbnails
st.header("Preprocessed symbols")
cols = st.columns(len(arrs))
for c, a, n in zip(cols, arrs, names):
    c.image(safe_image(a), caption=n, use_column_width=True)

# multilayer read for each symbol
st.header("Multi-layer reads (ranked hypotheses)")
for i, (arr, nm) in enumerate(zip(arrs, names)):
    ranked, explanation, dna = multilayer_read(arr, nm, pipeline_mode=("preload" if pipeline_mode=="preload (rules only)" else ("learn" if pipeline_mode=="learn (k-NN only)" else "hybrid")))
    st.subheader(f"{i+1}. {nm}")
    if ranked:
        for lab, score in ranked[:6]:
            st.write(f"- {lab}  — score {score:.2f}")
    else:
        st.write("_No hypotheses generated._")
    st.write("Explanation (features + top rule hits):")
    st.json(explanation)

    # human-in-loop: save mapping option
    label = st.text_input(f"Set mapping for {nm} (optional)", key=f"map_{i}")
    if st.button(f"Save mapping {nm}", key=f"save_map_{i}"):
        if label.strip():
            st.session_state.user_mappings[nm] = {"saved_label": label.strip(), "features": explanation["features"], "dna": dna.tolist()}
            save_json(MAPPINGS_FILE, st.session_state.user_mappings)
            st.success(f"Saved mapping {nm} -> {label.strip()}")
            record_event({"type":"mapping","symbol":nm,"label":label.strip()})
        else:
            st.info("Enter a non-empty label")

# Multi-symbol Harmonic Sequencer (item 1)
st.header("Multi-Symbol Harmonic Sequencer")
st.write("Select symbols and harmonic parameters; produce composite waveform + FFT + meaning summary")
sel_idxs = st.multiselect("Select indices to sequence", options=list(range(len(arrs))), default=list(range(min(3,len(arrs)))))
if len(sel_idxs) > 0:
    freq_map = {}
    for idx in sel_idxs:
        base_guess = 80 + features_list[idx]["lobe_count"]*30 + int(features_list[idx]["centroid_shift"]*200)
        freq_map[idx] = st.number_input(f"Base freq for {names[idx]}", min_value=20, max_value=2000, value=int(base_guess), key=f"bf_{idx}")
    harmonic_mult = st.slider("Harmonic multiplier max", 1, 8, 3)
    phase_offsets_deg = [st.slider(f"Phase° for {names[idx]}", 0, 360, 0, key=f"phseq_{idx}") for idx in sel_idxs]
    weights = [st.slider(f"Weight for {names[idx]}", 0.0, 3.0, 1.0, key=f"wseq_{idx}") for idx in sel_idxs]
    if st.button("Run harmonic sequence"):
        # build composite acoustic spectrum by summing gaussian peaks
        freqs = np.linspace(1,2000,2000)
        combined_spec = np.zeros_like(freqs)
        for idx, base in freq_map.items():
            for mult in range(1, harmonic_mult+1):
                peak = base * mult
                amp = 1.0 / (mult)
                combined_spec += weights[sel_idxs.index(idx)] * amp * np.exp(-0.5*((freqs - peak)/(peak*0.03+1e-6))**2)
        fig, ax = plt.subplots(figsize=(6,2))
        ax.plot(freqs, combined_spec)
        ax.set_xlim(0,800); ax.set_xlabel("Hz")
        st.pyplot(fig)
        # optical composite (simple weighted sum + phase roll)
        comp = np.zeros_like(arrs[0])
        for idx, w, ph in zip(sel_idxs, weights, phase_offsets_deg):
            comp += w * np.roll(arrs[idx], int(ph/360.0*arrs[idx].shape[1]), axis=1)
        st.image(safe_image(comp), caption="Optical composite", use_column_width=True)
        # meaning summary (run multilayer on composite)
        ranked_comp, expl_comp, dna_comp = multilayer_read(safe_image(comp), "composite", pipeline_mode=("hybrid" if pipeline_mode=="hybrid (recommended)" else ("preload" if pipeline_mode=="preload (rules only)" else "learn")))
        st.write("Composite hypotheses:")
        for lab, sc in ranked_comp[:6]:
            st.write(f"- {lab} — score {sc:.2f}")
        record_event({"type":"sequence_run","indices":sel_idxs,"weights":weights,"phases":phase_offsets_deg})

# AC/DC Frequency Sweep Engine (item 3)
st.header("AC/DC Frequency Sweep Engine")
sweep_mode = st.selectbox("Sweep mode", ["linear","log","chirp","burst"])
sweep_start = st.number_input("Start frequency (Hz)", 1, 1, 20)
sweep_stop = st.number_input("Stop frequency (Hz)", 1, 400, 400)
sweep_frames = st.slider("Sweep frames (GIF)", 8, 60, 24)
sweep_duration = st.slider("Sweep duration (s)", 1.0, 6.0, 2.0)
if st.button("Generate sweep GIF (acoustic)"):
    # build acoustic sweep by creating spec frames across frequencies
    frames = []
    freqs = np.linspace(1,2000,2000)
    for t in range(sweep_frames):
        if sweep_mode == "linear":
            fbase = sweep_start + (sweep_stop - sweep_start) * (t/(sweep_frames-1))
        elif sweep_mode == "log":
            fbase = np.exp(np.log(sweep_start) + (np.log(sweep_stop)-np.log(sweep_start))*(t/(sweep_frames-1)))
        elif sweep_mode == "chirp":
            fbase = sweep_start + (sweep_stop - sweep_start)*(t/(sweep_frames-1))**2
        else:  # burst
            fbase = sweep_start if (t < sweep_frames/2) else sweep_stop
        combined = np.zeros_like(freqs)
        for dna in dna_list[:min(4,len(dna_list))]:
            centroid = np.sum(np.arange(len(dna)) * dna) / (dna.sum()+1e-12)
            base = fbase * (0.5 + centroid)
            for mult in [1,2,3]:
                peak = base*mult
                combined += np.exp(-0.5*((freqs - peak)/(peak*0.03+1e-6))**2)
        im = np.zeros((128,512), dtype=np.uint8)
        env = combined[:800]; env = env - env.min()
        if env.max()>0:
            env = env / env.max()
        ys = (env * 127).astype(np.uint8)
        for x, y in enumerate(ys):
            im[127-y:, x] = 255
        frames.append(Image.fromarray(im).convert("L"))
    buf = io.BytesIO()
    frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=int(1000*sweep_duration/sweep_frames), loop=0)
    buf.seek(0)
    st.image(buf.getvalue())
    st.download_button("Download sweep GIF", buf.getvalue(), file_name="acoustic_sweep.gif", mime="image/gif")
    record_event({"type":"sweep","mode":sweep_mode,"start":sweep_start,"stop":sweep_stop})

# Symbol-to-Symbol Energy Flow Simulator (item 4)
st.header("Symbol-to-Symbol Energy Flow Simulator")
from_idx = st.selectbox("From symbol index", options=list(range(len(arrs))), format_func=lambda i: names[i])
to_idx = st.selectbox("To symbol index", options=list(range(len(arrs))), format_func=lambda i: names[i])
alpha_flow = st.slider("Transform strength (alpha)", 0.0, 3.0, 1.0)
if st.button("Simulate energy flow A→B"):
    trans = energy_flow_transform(arrs[from_idx], arrs[to_idx], alpha=alpha_flow)
    st.image(trans, caption=f"Transformed output (A→B)", use_column_width=True)
    # show comparison
    colA, colB, colC = st.columns(3)
    colA.image(safe_image(arrs[from_idx]), caption=f"A: {names[from_idx]}")
    colB.image(safe_image(arrs[to_idx]), caption=f"B: {names[to_idx]}")
    colC.image(trans, caption="A→B transformed")
    record_event({"type":"energy_flow","from":names[from_idx],"to":names[to_idx],"alpha":alpha_flow})

# Auto-crop & Multi-crop improved already implemented above (item 5)

# Enhanced Hypothesis report (item 6)
st.header("Export / Report")
if st.button("Export Features & DNA ZIP"):
    zipb = export_features_and_dna_zip(names, features_list, dna_list)
    st.download_button("Download export ZIP", zipb, file_name="indus_export.zip", mime="application/zip")
if st.button("Generate HTML Hypothesis Report"):
    rep = generate_html_report(names, features_list, dna_list, cosmic_list)
    st.download_button("Download HTML report", rep.encode('utf-8'), file_name="indus_hypothesis_report.html", mime="text/html")
    st.success("Download ready — open HTML and Save As PDF for a PDF copy.")

# Real-time oscilloscope & spectrogram (frame-based) (item 7)
st.header("Oscilloscope & Spectrogram (frame snapshots)")
# For display we synth a waveform from selected symbol DNA-derived frequency and show waveform + spectrogram
osc_idx = st.selectbox("Pick symbol for osc/spectrogram", options=list(range(len(arrs))), format_func=lambda i: names[i])
if st.button("Generate osc & spectrogram snapshot"):
    dna = dna_list[osc_idx]
    centroid = np.sum(np.arange(len(dna)) * dna) / (dna.sum()+1e-12)
    base_freq = 40 + centroid * 760
    sr = 22050; dur = 1.0
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    sig = np.zeros_like(t)
    for mult in [1,2,3,4][:4]:
        sig += (1.0/mult) * np.sin(2*np.pi*base_freq*mult*t)
    fig, axs = plt.subplots(2,1, figsize=(8,4))
    axs[0].plot(t[:1000], sig[:1000]); axs[0].set_title("Oscilloscope (time domain)")
    axs[1].specgram(sig, NFFT=1024, Fs=sr, noverlap=512); axs[1].set_title("Spectrogram")
    st.pyplot(fig)
    record_event({"type":"osc_snapshot","symbol":names[osc_idx],"base_freq":base_freq})

# Indus Grammar Engine (Experimental) (item 8)
st.header("Indus Grammar Engine (experimental)")
seq_input = st.text_input("Enter a sequence of indices (comma-separated) e.g. 0,1,2", value=",".join(str(i) for i in range(min(3,len(arrs)))))
if st.button("Analyze sequence grammar"):
    try:
        seq = [int(x.strip()) for x in seq_input.split(",") if x.strip()!='']
        # heuristics: operators tend to have high rot_harmonics, carriers high v_score, modifiers have centroid shifts
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
            roles.append((names[idx], role, f))
        st.write("Sequence roles (heuristic):")
        for r in roles:
            st.write(f"- {r[0]} → {r[1]}")
        # attempt simple syntactic reduction: operator + carrier -> compounded meaning
        st.write("Syntactic summary (heuristic):")
        if len(roles) >= 2:
            st.write(f"Primary: {roles[0][1]} | Secondary: {roles[1][1]}")
        record_event({"type":"grammar_analysis","seq":seq,"roles":[r[1] for r in roles]})
    except Exception as e:
        st.error(f"Parse error: {e}")

# Session log & persistence
st.header("Session log & persistence")
st.write("Recent events (last 20):")
st.write(st.session_state.session_log[-20:])
if st.button("Save session log to disk"):
    save_json(SESSION_FILE, st.session_state.session_log)
    st.success(f"Saved session log to {SESSION_FILE}")
if st.button("Clear session log"):
    st.session_state.session_log = []; save_json(SESSION_FILE, st.session_state.session_log); st.success("Cleared session log")

# Save mappings
if st.button("Save user mappings to disk"):
    save_json(MAPPINGS_FILE, st.session_state.user_mappings)
    st.success(f"Saved mappings to {MAPPINGS_FILE}")

st.success("All systems loaded. Use the UI above to run experiments — hybrid pipeline (preload rules + learn) is recommended.")
