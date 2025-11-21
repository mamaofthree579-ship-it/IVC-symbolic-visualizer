# app.py
# Indus Resonance Lab — Full + Visual Crop Editor + Rule Editor + Persistent Saves
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io, os, json, csv, math
from typing import List, Dict, Any

# -----------------------
# CONFIG / PERSIST PATHS
# -----------------------
WORKSPACE_SHEET_PATH = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"
PERSIST_DIR = "indus_state"
RULES_FILE = os.path.join(PERSIST_DIR, "rule_base.json")
CENTROIDS_FILE = os.path.join(PERSIST_DIR, "kmeans_centroids.json")
MAPPINGS_FILE = os.path.join(PERSIST_DIR, "user_mappings.json")
SESSION_LOG_FILE = os.path.join(PERSIST_DIR, "session_log.json")

os.makedirs(PERSIST_DIR, exist_ok=True)

# -----------------------
# DEFAULTS (loaded/saved)
# -----------------------
DEFAULT_RULES = [
    {"id":"rule_container","label":"Container/Storage","conditions":{"v_score_min":0.6,"h_score_max":0.6,"lobe_max":2},"confidence":0.85,"explanation":"vertical symmetry + few lobes"},
    {"id":"rule_flow","label":"Flow/Movement","conditions":{"h_score_min":0.45,"centroid_shift_min":0.03},"confidence":0.78,"explanation":"horizontal symmetry + centroid shift"},
    {"id":"rule_pair","label":"Duality/Pairing","conditions":{"lobe_min":3,"rot_harmonics_min":2},"confidence":0.88,"explanation":"multiple lobes and rotational harmonics"},
    {"id":"rule_celestial","label":"Cosmic/Celestial","conditions":{"rot_harmonics_min":4,"lobe_min":4},"confidence":0.92,"explanation":"rich rotational harmonic structure"},
    {"id":"rule_numeric","label":"Numeric/Count","conditions":{"lobe_max":1,"centroid_shift_max":0.02},"confidence":0.6,"explanation":"simple centered mark"}
]

# Helper to persist/load JSON safely
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# Load persisted states (or defaults)
persisted_rules = load_json(RULES_FILE, DEFAULT_RULES)
persisted_mappings = load_json(MAPPINGS_FILE, {})
persisted_centroids = load_json(CENTROIDS_FILE, {})
persisted_session_log = load_json(SESSION_LOG_FILE, [])

# Ensure session_state keys
if "rule_base" not in st.session_state:
    st.session_state.rule_base = persisted_rules
if "user_mappings" not in st.session_state:
    st.session_state.user_mappings = persisted_mappings
if "kmeans_centroids" not in st.session_state:
    st.session_state.kmeans_centroids = persisted_centroids
if "session_log" not in st.session_state:
    st.session_state.session_log = persisted_session_log

# -----------------------
# Utilities: image preprocess + normalizer
# -----------------------
def autocrop(img: Image.Image, tol: int = 10) -> Image.Image:
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

def load_and_normalize(img_file, target_size:int=256) -> np.ndarray:
    if isinstance(img_file, str):
        img = Image.open(img_file).convert("L")
    else:
        img = Image.open(img_file).convert("L")
    img = autocrop(img)
    w, h = img.size
    scale = target_size / max(w, h)
    img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    canvas = Image.new("L", (target_size, target_size), 255)
    offset = ((target_size - img.size[0])//2, (target_size - img.size[1])//2)
    canvas.paste(img, offset)
    arr = np.array(canvas).astype(np.float32)
    arr = (arr - arr.min()) / max(1e-12, (arr.max() - arr.min()))
    return arr

def safe_image(arr: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(arr)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return a

# -----------------------
# Geometry extraction
# -----------------------
def geometry_features_from_image(arr: np.ndarray) -> Dict[str,Any]:
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

    lap = np.abs(np.fft.ifft2((np.fft.fft2(arr) * (-4 * (np.sin(np.pi * (xx/N))**2 + np.sin(np.pi * (yy/N))**2)))))
    roughness = float(np.var(np.abs(lap)))

    return {
        "lobe_count": lobe_count,
        "v_score": round(v_score,4),
        "h_score": round(h_score,4),
        "rot_harmonics": rot_harmonics,
        "centroid_shift": round(centroid_shift,4),
        "roughness": round(roughness,8),
        "mag": mag.tolist()  # store as list for JSON-friendly preview if needed
    }

def geometry_code(features: Dict[str,Any]) -> str:
    return f"G{features['lobe_count']}-V{features['v_score']:.2f}-H{features['h_score']:.2f}-R{features['rot_harmonics']}"

# -----------------------
# Rule engine helpers
# -----------------------
def rule_matches(features: Dict[str,Any], rule: Dict[str,Any]) -> (bool, Dict[str,Any]):
    c = rule.get("conditions", {})
    detail = {}
    # checks
    if "v_score_min" in c:
        detail["v_score"] = (features["v_score"], c["v_score_min"])
        if features["v_score"] < c["v_score_min"]:
            return False, detail
    if "v_score_max" in c:
        detail["v_score"] = (features["v_score"], c["v_score_max"])
        if features["v_score"] > c["v_score_max"]:
            return False, detail
    if "h_score_min" in c:
        detail["h_score"] = (features["h_score"], c["h_score_min"])
        if features["h_score"] < c["h_score_min"]:
            return False, detail
    if "h_score_max" in c:
        detail["h_score"] = (features["h_score"], c["h_score_max"])
        if features["h_score"] > c["h_score_max"]:
            return False, detail
    if "lobe_min" in c:
        detail["lobe_count"] = (features["lobe_count"], c["lobe_min"])
        if features["lobe_count"] < c["lobe_min"]:
            return False, detail
    if "lobe_max" in c:
        detail["lobe_count"] = (features["lobe_count"], c["lobe_max"])
        if features["lobe_count"] > c["lobe_max"]:
            return False, detail
    if "rot_harmonics_min" in c:
        detail["rot_harmonics"] = (features["rot_harmonics"], c["rot_harmonics_min"])
        if features["rot_harmonics"] < c["rot_harmonics_min"]:
            return False, detail
    if "centroid_shift_min" in c:
        detail["centroid_shift"] = (features["centroid_shift"], c["centroid_shift_min"])
        if features["centroid_shift"] < c["centroid_shift_min"]:
            return False, detail
    if "centroid_shift_max" in c:
        detail["centroid_shift"] = (features["centroid_shift"], c["centroid_shift_max"])
        if features["centroid_shift"] > c["centroid_shift_max"]:
            return False, detail
    return True, detail

def fallback_predict(features: Dict[str,Any]) -> List[tuple]:
    fv = np.array([
        min(features["lobe_count"],8)/8.0,
        features["v_score"],
        features["h_score"],
        min(features["rot_harmonics"],8)/8.0,
        features["centroid_shift"]
    ])
    PROTOTYPES = {
        "Container/Storage": np.array([1.0,0.8,0.2,0.5,0.02]),
        "Flow/Movement": np.array([0.3,0.2,0.8,0.7,0.12]),
        "Duality/Pairing": np.array([0.7,0.4,0.4,0.9,0.04]),
        "Cosmic/Celestial": np.array([1.0,0.5,0.5,1.0,0.01]),
        "Numeric/Count": np.array([0.05,0.95,0.95,0.05,0.005])
    }
    dists = {}
    for label, proto in PROTOTYPES.items():
        d = np.linalg.norm(fv - proto)
        dists[label] = float(d)
    maxd = max(dists.values()) if dists else 1.0
    sims = {k: 1 - (v/(maxd+1e-12)) for k,v in dists.items()}
    sorted_sims = sorted(sims.items(), key=lambda x:-x[1])
    return sorted_sims

# -----------------------
# kmeans (numpy)
# -----------------------
def kmeans(X: np.ndarray, k:int=3, iters:int=100, seed:int=0):
    if X.shape[0] == 0:
        return np.array([]), np.array([])
    rng = np.random.RandomState(seed)
    if X.shape[0] < k:
        k = X.shape[0]
    centroids = X[rng.choice(X.shape[0], k, replace=False)]
    for _ in range(iters):
        dists = np.linalg.norm(X[:,None,:] - centroids[None,:,:], axis=2)
        labels = np.argmin(dists, axis=1)
        newc = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centroids[i] for i in range(k)])
        if np.allclose(newc, centroids):
            break
        centroids = newc
    return labels, centroids

# -----------------------
# session recorder + persist helpers
# -----------------------
def record_event(evt: Dict[str,Any]):
    st.session_state.session_log.append(evt)
    save_json(SESSION_LOG_FILE, st.session_state.session_log)

def persist_rules():
    save_json(RULES_FILE, st.session_state.rule_base)

def persist_mappings():
    save_json(MAPPINGS_FILE, st.session_state.user_mappings)

def persist_centroids(centroids_obj):
    save_json(CENTROIDS_FILE, centroids_obj)

# -----------------------
# Audio synth (small WAV buffer)
# -----------------------
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

# -----------------------
# UI
# -----------------------
st.header("1) Upload & Visual Crop Editor")
uploaded = st.file_uploader("Upload one or more symbol images (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)

# visual cropper: show the first uploaded image (or workspace sheet) and allow cropping via sliders
st.markdown("**Interactive Crop Editor** — pick image, adjust crop rectangle using sliders and preview crop. Use this to trim whitespace or focus the glyph before analysis.")
sample_for_crop = None
if uploaded and len(uploaded) > 0:
    sample_for_crop = uploaded[0]
elif os.path.exists(WORKSPACE_SHEET_PATH):
    sample_for_crop = WORKSPACE_SHEET_PATH

if sample_for_crop:
    if isinstance(sample_for_crop, str):
        img_for_crop = Image.open(sample_for_crop).convert("RGB")
    else:
        sample_for_crop.seek(0)
        img_for_crop = Image.open(sample_for_crop).convert("RGB")
    w, h = img_for_crop.size
    st.write(f"Image for cropping: {getattr(sample_for_crop,'name', os.path.basename(WORKSPACE_SHEET_PATH))} — original size {w}×{h}")
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(img_for_crop, caption="Original image", use_column_width=True)
    # default crop values
    left = st.slider("left (px)", 0, w-1, 0)
    top = st.slider("top (px)", 0, h-1, 0)
    right = st.slider("right (px)", 1, w, w)
    bottom = st.slider("bottom (px)", 1, h, h)
    # button to apply crop and add to uploaded list as a BytesIO so it flows into the pipeline
    if st.button("Apply crop & add to processing list"):
        try:
            crop = img_for_crop.crop((left, top, right, bottom))
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            buf.seek(0)
            # append new crop to uploaded list in session for processing convenience
            if "manual_crops" not in st.session_state:
                st.session_state.manual_crops = []
            st.session_state.manual_crops.append(buf)
            st.success("Crop saved to session (appears alongside uploaded files).")
        except Exception as e:
            st.error(f"Crop failed: {e}")

# combine uploaded + manual crops for processing
processing_list = []
if uploaded:
    processing_list.extend(uploaded)
if "manual_crops" in st.session_state:
    processing_list.extend(st.session_state.manual_crops)

st.markdown("---")
st.header("2) Analysis & Meaning Inference")

if len(processing_list) == 0:
    st.info("Upload or create a crop to start analyzing symbols.")
else:
    # load, normalize & analyze all items in processing_list
    analyses = []
    for item in processing_list:
        try:
            arr = load_and_normalize(item, target_size=256)
            features = geometry_features_from_image(arr)
            code = geometry_code(features)
            # rule-based matches
            matches = []
            for r in st.session_state.rule_base:
                ok, detail = rule_matches(features, r)
                if ok:
                    matches.append({"label":r["label"], "confidence":r["confidence"], "explanation":r["explanation"], "rule_id":r["id"], "detail":detail})
            fallback = fallback_predict(features)
            analyses.append({"name": getattr(item, "name", "uploaded_crop"), "arr":arr, "features":features, "code":code, "matches":matches, "fallback":fallback})
            # record
            record_event({"type":"analyze","name":getattr(item,"name","uploaded"), "features":features})
        except Exception as e:
            st.warning(f"Skipping one item due to load error: {e}")

    # present analyses
    for rec in analyses:
        st.markdown("----")
        st.subheader(f"Symbol: {rec['name']}")
        st.image(safe_image(rec["arr"]), width=220)
        st.write("Geometry code:", rec["code"])
        st.write("Features (summary):", {k:v for k,v in rec["features"].items() if k!="mag"})
        st.write("Rule matches:")
        if rec["matches"]:
            for m in rec["matches"]:
                st.write(f"- {m['label']} (confidence {m['confidence']:.2f})")
                st.caption(m["explanation"])
        else:
            st.write("_No rule matched_")
        st.write("Fallback similarities (top 4):")
        for lab, sc in rec["fallback"][:4]:
            st.write(f"- {lab}: {sc:.2f}")
        # save mapping UI
        label_input = st.text_input(f"Save mapping label for {rec['name']}", key=f"label_{rec['name']}")
        if st.button(f"Save mapping for {rec['name']}", key=f"save_{rec['name']}"):
            st.session_state.user_mappings[rec['name']] = {"saved_label": label_input, "features": rec["features"]}
            persist_mappings()
            st.success(f"Saved mapping {rec['name']} → {label_input}")
            record_event({"type":"mapping","symbol":rec['name'],"label":label_input})
    # Composite & sequence on analyzed images
    if len(analyses) >= 2:
        st.markdown("---")
        st.subheader("Harmonic Composite (optical)")
        w1 = st.slider("Weight symbol A", 0.0, 3.0, 1.0, key="w1")
        w2 = st.slider("Weight symbol B", 0.0, 3.0, 1.0, key="w2")
        phase_px = st.slider("Phase shift (px roll)", 0, 128, 0)
        A = analyses[0]["arr"]; B = analyses[1]["arr"]
        composite = safe_image(w1*A + w2*np.roll(B, int(phase_px), axis=1))
        st.image(composite, caption="Composite", use_column_width=True)
        if st.button("Save composite event"):
            record_event({"type":"composite","weights":[w1,w2],"phase_px":int(phase_px)})

    if len(analyses) >= 3:
        st.markdown("---")
        st.subheader("Sequence Engine")
        phase_per = st.slider("Phase per symbol (px roll)", 0, 32, 2)
        out = np.zeros_like(analyses[0]["arr"])
        for i, rec in enumerate(analyses):
            out += np.roll(rec["arr"], int(i*phase_per), axis=0)
        st.image(safe_image(out), caption="Sequence output")
        if st.button("Save sequence event"):
            record_event({"type":"sequence","n":len(analyses),"phase_per":int(phase_per)})

    st.markdown("---")
    st.subheader("Audio Synthesizer (based on first symbol)")
    base_guess = 120 + analyses[0]["features"]["lobe_count"]*30 + int(analyses[0]["features"]["centroid_shift"]*200)
    base_freq = st.slider("Base frequency (Hz)", 20, 1200, base_guess)
    n_harm = st.slider("Number of harmonics", 1, 8, 3)
    harm_mults = [i+1 for i in range(n_harm)]
    harm_amps = [1.0/(i+1) for i in range(n_harm)]
    dur = st.slider("Duration (s)", 0.5, 5.0, 1.5)
    if st.button("Play tone"):
        wav = synth_tone(base_freq, duration=dur, harmonics=harm_mults, amps=harm_amps)
        st.audio(wav, format="audio/wav")
        record_event({"type":"audio","freq":base_freq,"harmonics":harm_mults,"duration":dur})

    st.markdown("---")
    st.subheader("Clustering (k-means) on current batch")
    X = []
    names = []
    for rec in analyses:
        f = rec["features"]
        vec = np.array([min(f["lobe_count"],8)/8.0, f["v_score"], f["h_score"], min(f["rot_harmonics"],8)/8.0, f["centroid_shift"]])
        X.append(vec); names.append(rec["name"])
    if len(X) >= 2:
        X = np.vstack(X)
        k = st.slider("k clusters", 2, min(8, X.shape[0]), 2)
        labels, cents = kmeans(X, k=k, iters=200)
        st.write("Assignments:")
        for n, lab in zip(names, labels):
            st.write(f"- {n}: cluster {int(lab)}")
        # persist centroids
        persist_centroids({"centroids": cents.tolist(), "names": names})
        st.session_state.kmeans_centroids = {"centroids": cents.tolist(), "names": names}
        record_event({"type":"kmeans","k":int(k),"labels":labels.tolist()})
        if st.button("Export centroids JSON"):
            save_json(CENTROIDS_FILE, st.session_state.kmeans_centroids)
            st.success(f"Saved centroids to {CENTROIDS_FILE}")
    else:
        st.info("Need at least 2 symbols to cluster.")

st.markdown("---")
st.header("3) Rule Editor & Persistence")

st.write("Edit rule base below. Changes persist to disk immediately (in `indus_state/rule_base.json`).")
# display current rules
for i, rule in enumerate(st.session_state.rule_base):
    st.markdown(f"**Rule #{i+1} — {rule['label']}** (id: {rule['id']})")
    with st.expander("Edit rule"):
        lab = st.text_input(f"Label #{i}", value=rule["label"], key=f"lab_{i}")
        conf = st.number_input(f"Confidence #{i}", min_value=0.0, max_value=1.0, value=float(rule.get("confidence",0.6)), key=f"conf_{i}")
        cond = rule.get("conditions", {})
        vmin = st.number_input(f"v_score_min_{i}", min_value=0.0, max_value=1.0, value=float(cond.get("v_score_min",0.0)), key=f"vmin_{i}")
        vmax = st.number_input(f"v_score_max_{i}", min_value=0.0, max_value=1.0, value=float(cond.get("v_score_max",1.0)), key=f"vmax_{i}")
        hmin = st.number_input(f"h_score_min_{i}", min_value=0.0, max_value=1.0, value=float(cond.get("h_score_min",0.0)), key=f"hmin_{i}")
        hmax = st.number_input(f"h_score_max_{i}", min_value=0.0, max_value=1.0, value=float(cond.get("h_score_max",1.0)), key=f"hmax_{i}")
        lmin = st.number_input(f"lobe_min_{i}", min_value=0, max_value=50, value=int(cond.get("lobe_min",0)), key=f"lmin_{i}")
        lmax = st.number_input(f"lobe_max_{i}", min_value=0, max_value=50, value=int(cond.get("lobe_max",50)), key=f"lmax_{i}")
        chg = st.button(f"Save changes to rule #{i}", key=f"save_rule_{i}")
        if chg:
            new_cond = {}
            if vmin > 0: new_cond["v_score_min"] = float(vmin)
            if vmax < 1.0: new_cond["v_score_max"] = float(vmax)
            if hmin > 0: new_cond["h_score_min"] = float(hmin)
            if hmax < 1.0: new_cond["h_score_max"] = float(hmax)
            if lmin > 0: new_cond["lobe_min"] = int(lmin)
            if lmax < 50: new_cond["lobe_max"] = int(lmax)
            st.session_state.rule_base[i]["label"] = lab
            st.session_state.rule_base[i]["confidence"] = float(conf)
            st.session_state.rule_base[i]["conditions"] = new_cond
            persist_rules()
            st.success("Rule updated and persisted.")

    if st.button(f"Delete rule #{i}", key=f"del_{i}"):
        st.session_state.rule_base.pop(i)
        persist_rules()
        st.experimental_rerun()

st.markdown("### Add a new rule")
new_label = st.text_input("New rule label", key="new_rule_label")
new_conf = st.number_input("confidence", min_value=0.0, max_value=1.0, value=0.6, key="new_conf")
new_vmin = st.number_input("v_score_min", min_value=0.0, max_value=1.0, value=0.0, key="new_vmin")
new_hmax = st.number_input("h_score_max", min_value=0.0, max_value=1.0, value=1.0, key="new_hmax")
new_lmax = st.number_input("lobe_max", min_value=0, max_value=50, value=10, key="new_lmax")
if st.button("Add new rule"):
    if new_label.strip():
        new_rule = {"id":f"r_{len(st.session_state.rule_base)+1}","label":new_label,"conditions":{},"confidence":float(new_conf),"explanation":"user-added"}
        if new_vmin > 0: new_rule["conditions"]["v_score_min"] = float(new_vmin)
        if new_hmax < 1.0: new_rule["conditions"]["h_score_max"] = float(new_hmax)
        if new_lmax < 50: new_rule["conditions"]["lobe_max"] = int(new_lmax)
        st.session_state.rule_base.append(new_rule)
        persist_rules()
        st.success("New rule added and persisted.")
    else:
        st.error("Provide a label.")

st.markdown("---")
st.header("Persistence / Exports")
st.write("Saved files (in working directory):")
st.write(f"- Rules JSON: {RULES_FILE}")
st.write(f"- KMeans centroids JSON: {CENTROIDS_FILE}")
st.write(f"- User mappings JSON: {MAPPINGS_FILE}")
st.write(f"- Session log JSON: {SESSION_LOG_FILE}")

if st.button("Export current mappings to JSON"):
    persist_mappings()
    st.success(f"Saved mappings to {MAPPINGS_FILE}")

if st.button("Export session log to JSON"):
    save_json(SESSION_LOG_FILE, st.session_state.session_log)
    st.success(f"Saved session log to {SESSION_LOG_FILE}")

st.markdown("---")
st.caption("Workspace sheet path (local):")
st.code(WORKSPACE_SHEET_PATH)
