# app.py
# Indus Resonance Lab — Full Feature Set
# - auto-crop & normalize
# - FFT & geometry features
# - meaning inference rules + higher-order layer
# - session recorder (JSON) + export
# - audio resonance synth (tones + harmonic stacks)
# - k-means clustering (NumPy implementation)
# - batch-run & optional sheet split from workspace path
#
# Note: the app expects your workspace sample sheet at:
# /mnt/data/A_digital_vector_image_displays_three_black_Indus_.png
# (if present the UI will let you auto-split it)

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import json, os, io, base64, math
from typing import List, Dict, Any

st.set_page_config(page_title="Indus Resonance Lab — Full", layout="wide")
st.title("Indus Symbol Resonance Lab — Full Suite")

# ---------------------------
# Utilities: image preprocess
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
    else:
        return img

def load_and_normalize(img_file, target_size:int=256) -> np.ndarray:
    """Accepts filepath or uploaded file-like; returns float32 array 0..1 of size target_size^2"""
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

# ---------------------------
# Geometry extraction
# ---------------------------
def geometry_features_from_image(arr: np.ndarray) -> Dict[str,Any]:
    # arr expected 0..1 float, shape (N,N)
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
    lobe_count = int((hist > (0.25 * hist.max())).sum()) if hist.max()>0 else 0

    # rotational harmonics
    fft1d = np.abs(np.fft.fft(hist))
    rot_harmonics = int((fft1d > (0.2 * fft1d.max())).sum()) if fft1d.max()>0 else 0

    # centroid shift
    M = norm.sum() + 1e-12
    cyc = np.sum(norm * yy) / M
    cxc = np.sum(norm * xx) / M
    centroid_shift = float(np.sqrt((cxc - cx)**2 + (cyc - cy)**2) / (N/2.0))

    # fractal-ish roughness via Laplacian variance (higher -> more complex/higher-harmonic)
    lap = np.abs(np.fft.ifft2((np.fft.fft2(arr) * (-4 * (np.sin(np.pi * (xx/N))**2 + np.sin(np.pi * (yy/N))**2)))))
    roughness = float(np.var(np.abs(lap)))

    features = {
        "lobe_count": lobe_count,
        "v_score": round(v_score,4),
        "h_score": round(h_score,4),
        "rot_harmonics": rot_harmonics,
        "centroid_shift": round(centroid_shift,4),
        "roughness": round(roughness,8),
        "mag": mag  # keep for visualization
    }
    return features

def geometry_code(features: Dict[str,Any]) -> str:
    return f"G{features['lobe_count']}-V{features['v_score']:.2f}-H{features['h_score']:.2f}-R{features['rot_harmonics']}"

# ---------------------------
# Meaning inference: rules + higher-order layer
# ---------------------------
DEFAULT_RULES = [
    {"id":"rule_container","label":"Container/Storage","conditions":{"v_score_min":0.6,"h_score_max":0.6,"lobe_max":2},"confidence":0.85,"explanation":"vertical symmetry + few lobes"},
    {"id":"rule_flow","label":"Flow/Movement","conditions":{"h_score_min":0.45,"centroid_shift_min":0.03},"confidence":0.78,"explanation":"horizontal symmetry + centroid shift"},
    {"id":"rule_pair","label":"Duality/Pairing","conditions":{"lobe_min":3,"rot_harmonics_min":2},"confidence":0.88,"explanation":"multiple lobes and rotational harmonics"},
    {"id":"rule_celestial","label":"Cosmic/Celestial","conditions":{"rot_harmonics_min":4,"lobe_min":4},"confidence":0.92,"explanation":"rich rotational harmonic structure"},
    {"id":"rule_numeric","label":"Numeric/Count","conditions":{"lobe_max":1,"centroid_shift_max":0.02},"confidence":0.6,"explanation":"simple centered mark"}
]

PROTOTYPES = {
    "Container/Storage": np.array([1.0,0.8,0.2,0.5,0.02]),
    "Flow/Movement": np.array([0.3,0.2,0.8,0.7,0.12]),
    "Duality/Pairing": np.array([0.7,0.4,0.4,0.9,0.04]),
    "Cosmic/Celestial": np.array([1.0,0.5,0.5,1.0,0.01]),
    "Numeric/Count": np.array([0.05,0.95,0.95,0.05,0.005])
}

def rule_matches(features: Dict[str,Any], rule: Dict[str,Any]) -> (bool, Dict[str,Any]):
    c = rule.get("conditions",{})
    detail = {}
    def ch(k, op, val):
        detail[k] = (features.get(k), val)
        if op == 'lt':
            return features.get(k) < val
        elif op == 'gt':
            return features.get(k) > val
        return True
    # checks
    if "v_score_min" in c and features["v_score"] < c["v_score_min"]:
        return False, detail
    if "v_score_max" in c and features["v_score"] > c["v_score_max"]:
        return False, detail
    if "h_score_min" in c and features["h_score"] < c["h_score_min"]:
        return False, detail
    if "h_score_max" in c and features["h_score"] > c["h_score_max"]:
        return False, detail
    if "lobe_min" in c and features["lobe_count"] < c["lobe_min"]:
        return False, detail
    if "lobe_max" in c and features["lobe_count"] > c["lobe_max"]:
        return False, detail
    if "rot_harmonics_min" in c and features["rot_harmonics"] < c["rot_harmonics_min"]:
        return False, detail
    if "centroid_shift_min" in c and features["centroid_shift"] < c["centroid_shift_min"]:
        return False, detail
    if "centroid_shift_max" in c and features["centroid_shift"] > c["centroid_shift_max"]:
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
    dists = {}
    for label, proto in PROTOTYPES.items():
        d = np.linalg.norm(fv - proto)
        dists[label] = float(d)
    maxd = max(dists.values()) if dists else 1.0
    sims = {k: 1 - (v/(maxd+1e-12)) for k,v in dists.items()}
    sorted_sims = sorted(sims.items(), key=lambda x:-x[1])
    return sorted_sims

# ---------------------------
# k-means clustering (NumPy)
# ---------------------------
def kmeans(X: np.ndarray, k:int=3, iters:int=100, seed:int=0):
    if X.shape[0] == 0:
        return np.array([]), np.array([])
    rng = np.random.RandomState(seed)
    centroids = X[rng.choice(X.shape[0], k, replace=False)]
    for _ in range(iters):
        dists = np.linalg.norm(X[:,None,:] - centroids[None,:,:], axis=2)
        labels = np.argmin(dists, axis=1)
        newc = np.array([X[labels==i].mean(axis=0) if np.any(labels==i) else centroids[i] for i in range(k)])
        if np.allclose(newc, centroids):
            break
        centroids = newc
    return labels, centroids

# ---------------------------
# session recorder
# ---------------------------
if "session_log" not in st.session_state:
    st.session_state.session_log = []

def record_event(event: Dict[str,Any]):
    st.session_state.session_log.append(event)

# ---------------------------
# audio synth (simple): generate wav bytes from freq list/harmonics
# ---------------------------
def synth_tone(freq: float, duration: float=2.0, sr:int=22050, harmonics: List[float]=None, amps:List[float]=None):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    signal = np.zeros_like(t)
    if harmonics is None:
        harmonics = [1.0]
    if amps is None:
        amps = [1.0]*len(harmonics)
    for h, a in zip(harmonics, amps):
        signal += a * np.sin(2*np.pi*freq*h*t)
    # normalize to int16
    sig = signal / (np.max(np.abs(signal)) + 1e-12)
    samples = np.int16(sig * 32767)
    # write WAV bytes
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
# UI layout
# ---------------------------
colA, colB = st.columns([2,1])

with colA:
    st.header("Upload / Auto-split / Analyze")
    uploaded = st.file_uploader("Upload symbol images (JPG/PNG) — multiple allowed", type=["jpg","jpeg","png"], accept_multiple_files=True)
    st.markdown("Or auto-split the workspace sheet if present (developer path):")
    SAMPLE_SHEET = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"
    if os.path.exists(SAMPLE_SHEET):
        if st.button("Auto-split workspace sheet into three symbols"):
            try:
                sheet = Image.open(SAMPLE_SHEET).convert("L")
                w,h = sheet.size
                parts = []
                for i in range(3):
                    left = int(i*w/3); right = int((i+1)*w/3)
                    crop = sheet.crop((left,0,right,h)).resize((256,256))
                    parts.append(np.array(crop).astype(np.float32)/255.0)
                # place into uploaded-like list
                uploaded = [io.BytesIO() for _ in range(3)]
                for buf,img_arr in zip(uploaded, parts):
                    im = Image.fromarray((img_arr*255).astype(np.uint8))
                    im.save(buf, format='PNG'); buf.seek(0)
                st.success("Auto-split complete — analyze below.")
            except Exception as e:
                st.error(f"Auto-split failed: {e}")

    if uploaded:
        imgs = []
        names = []
        for f in uploaded:
            try:
                arr = load_and_normalize(f, target_size=256)
                imgs.append(arr); 
                names.append(getattr(f, "name", "uploaded"))
            except Exception as e:
                st.warning(f"Could not load one file: {e}")

        st.subheader("Preprocessed symbols")
        cols = st.columns(len(imgs))
        for c, a, n in zip(cols, imgs, names):
            c.image(safe_image(a), caption=n, use_column_width=True)

        # per-symbol analysis
        results = []
        for idx, (arr, nm) in enumerate(zip(imgs, names)):
            st.markdown(f"---\n### Symbol {idx+1} — {nm}")
            features = geometry_features_from_image(arr)
            code = geometry_code(features)
            st.write("Geometry code:", code)
            st.write("Features:", {k:v for k,v in features.items() if k!="mag"})
            st.image(safe_image(features["mag"]), caption="FFT (log mag)", use_column_width=True)

            # inference
            matches = []
            rejections = []
            for r in DEFAULT_RULES:
                ok, detail = rule_matches(features, r)
                if ok:
                    matches.append({"label":r["label"], "confidence":r["confidence"], "explanation":r["explanation"], "rule_id":r["id"]})
                else:
                    rejections.append({"rule_id":r["id"], "detail": detail})
            fallback = fallback_predict(features)

            st.subheader("Meaning Inference")
            if matches:
                for m in matches:
                    st.write(f"- **{m['label']}** (rule confidence {m['confidence']:.2f})")
                    st.caption(m['explanation'])
            else:
                st.write("_No rule matched explicitly._")

            st.write("Fallback similarities (top 5):")
            for lab, sc in fallback[:5]:
                st.write(f"- {lab} — score {sc:.2f}")

            # higher-order checks
            ho = []
            # fractal/roughness -> ritual/complex
            if features["roughness"] > 1e-5:
                ho.append(("Complex/Holographic", f"roughness {features['roughness']:.8f}"))
            # centroid far-off -> mark of locality or direction
            if features["centroid_shift"] > 0.15:
                ho.append(("Directional/Locality", f"centroid_shift {features['centroid_shift']:.3f}"))
            # rotational harmonic richness -> cosmic/high-order
            if features["rot_harmonics"] >= 4:
                ho.append(("Cosmic/High-order", f"rot_harmonics {features['rot_harmonics']}"))
            st.write("Higher-order signals:")
            for tag,info in ho:
                st.write(f"- {tag}: {info}")

            # store result object
            rec = {"name":nm, "features":features, "rules":matches, "fallback":fallback[:5], "higher_order":ho}
            results.append(rec)

            # record event to session log
            record_event({"type":"analyze","symbol":nm,"features":features,"inference":rec})

        # allow human-in-the-loop label saving
        st.subheader("Human-in-the-loop: Save labels")
        for rec in results:
            nm = rec["name"]
            label = st.text_input(f"Label for {nm}", key=f"label_{nm}")
            if st.button(f"Save mapping for {nm}", key=f"save_{nm}"):
                st.session_state.session_log.append({"type":"mapping","symbol":nm,"label":label,"features":rec["features"]})
                st.success(f"Saved mapping {nm} -> {label}")

        # composite controls
        if len(imgs) >= 2:
            st.subheader("Harmonic Composer (optical sum)")
            w1 = st.slider("Weight A", 0.0, 3.0, 1.0)
            w2 = st.slider("Weight B", 0.0, 3.0, 1.0)
            phase = st.slider("Phase shift (px roll)", 0, 64, 0)
            comp = safe_image(w1*imgs[0] + w2*np.roll(imgs[1], phase, axis=1))
            st.image(comp, caption="Composite", use_column_width=True)
            record_event({"type":"composite","weights":[w1,w2],"phase":int(phase)})

        # sequence engine
        if len(imgs) >= 3:
            st.subheader("Sequence Engine")
            phase_per = st.slider("Phase per symbol (px roll)", 0, 32, 2)
            out = np.zeros_like(imgs[0])
            for i, a in enumerate(imgs):
                out += np.roll(a, int(i*phase_per), axis=0)
            st.image(safe_image(out), caption="Sequence Output", use_column_width=True)
            record_event({"type":"sequence","n":len(imgs),"phase_per":int(phase_per)})

        # audio synth controls (use first symbol's base frequency from heuristic)
        st.subheader("Audio Resonance Synthesizer")
        # heuristic base freq: map centroid shift and lobe_count to 20..500 Hz
        avg_base = 120 + (results[0]["features"]["lobe_count"] * 30) + int(results[0]["features"]["centroid_shift"] * 200)
        base_freq = st.slider("Base frequency (Hz)", 20, 1000, int(avg_base))
        n_harm = st.slider("Number of harmonics", 1, 6, 3)
        harm_mults = [i+1 for i in range(n_harm)]
        harm_amps = [1.0/(i+1) for i in range(n_harm)]
        dur = st.slider("Duration (s)", 0.5, 6.0, 2.0)
        if st.button("Play synthesized resonance"):
            wav = synth_tone(base_freq, duration=dur, harmonics=harm_mults, amps=harm_amps)
            st.audio(wav, format="audio/wav")
            record_event({"type":"audio_play","base_freq":base_freq,"harmonics":harm_mults,"duration":dur})

        # clustering: build feature matrix of analyzed images
        st.subheader("Clustering (k-means) on analyzed symbols")
        X = []
        names_X = []
        for rec in results:
            f = rec["features"]
            vec = np.array([min(f["lobe_count"],8)/8.0, f["v_score"], f["h_score"], min(f["rot_harmonics"],8)/8.0, f["centroid_shift"]])
            X.append(vec); names_X.append(rec["name"])
        X = np.vstack(X) if X else np.zeros((0,5))
        if X.shape[0] >= 2:
            k = st.slider("k clusters", 2, min(8, X.shape[0]), 2)
            labels, cents = kmeans(X, k=k, iters=100)
            st.write("Cluster assignments:")
            for n, lab in zip(names_X, labels):
                st.write(f"- {n} → cluster {int(lab)}")
            record_event({"type":"cluster","k":int(k),"labels":labels.tolist()})
        else:
            st.info("Need at least 2 analyzed symbols to cluster.")

with colB:
    st.header("Session Recorder & Tools")
    st.write("Events recorded this session:")
    st.write(f"Total events: {len(st.session_state.session_log)}")
    if st.session_state.session_log:
        st.json(st.session_state.session_log[-20:])
    if st.button("Download session log (JSON)"):
        out = json.dumps(st.session_state.session_log, indent=2)
        b = out.encode('utf-8')
        st.download_button("⬇ Download session JSON", b, file_name="indus_session_log.json", mime="application/json")
    if st.button("Clear session log"):
        st.session_state.session_log = []
        st.success("Cleared session log")

    st.markdown("---")
    st.header("Batch Inference")
    folder = st.text_input("Batch folder path (relative)", value="sample_data")
    if st.button("Run batch inference"):
        if not os.path.exists(folder):
            st.error("Folder not found")
        else:
            rows = []
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith(('.jpg','.jpeg','.png')):
                    p = os.path.join(folder, fn)
                    try:
                        arr = load_and_normalize(p)
                        feats = geometry_features_from_image(arr)
                        fallback = fallback_predict(feats)
                        # rule merge
                        hits = [r["label"] for r in DEFAULT_RULES if rule_matches(feats, r)[0]]
                        combined = {}
                        for lab, score in fallback:
                            combined[lab] = 0.3 * score
                        for h in hits:
                            conf = next((r["confidence"] for r in DEFAULT_RULES if r["label"]==h), 0.6)
                            combined[h] = combined.get(h,0.0) + 0.7*conf
                        ranked = sorted(combined.items(), key=lambda x:-x[1])
                        top = ranked[0] if ranked else ("",0.0)
                        rows.append({"file":fn,"geometry":geometry_code(feats),"top_label":top[0],"score":round(top[1],3)})
                    except Exception as e:
                        rows.append({"file":fn,"error":str(e)})
            # produce CSV bytes
            import csv, tempfile
            tbuf = io.StringIO()
            writer = csv.writer(tbuf)
            writer.writerow(["file","geometry","top_label","score"])
            for r in rows:
                writer.writerow([r.get("file"), r.get("geometry",""), r.get("top_label",""), r.get("score","")])
            tbuf.seek(0)
            st.download_button("Download batch CSV", tbuf.getvalue().encode('utf-8'), file_name="batch_inference.csv", mime="text/csv")
            st.write("Preview:")
            st.dataframe(rows)

st.caption("Notes: this app is purposely dependency-light. The kmeans implementation is NumPy-only. The audio synth writes a small WAV buffer and plays it with Streamlit's st.audio. The workspace sample sheet path used (if present) is:")
st.code(SAMPLE_SHEET)
