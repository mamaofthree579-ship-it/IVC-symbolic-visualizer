# app.py
# Indus Resonance System + Meaning Inference Engine + kNN + Explain + Batch-run
import streamlit as st
import numpy as np
from PIL import Image
import os
import io
import csv
import json
from typing import Dict, Any, List, Tuple

st.set_page_config(page_title="Indus Resonance • Meaning + kNN + Batch", layout="wide")
st.title("Indus Symbol Resonance — Meaning Engine, k-NN Learner, Explain & Batch")

# -------------------------
# Utilities: safe normalizer + loader
# -------------------------
def safe_image(arr):
    arr = np.nan_to_num(arr)
    arr = arr - arr.min()
    maxv = arr.max()
    if maxv > 0:
        arr = arr / maxv
    return arr

def load_and_normalize(img_file, target_size=256):
    # img_file can be a path or a file-like object (Streamlit upload)
    if isinstance(img_file, str):
        img = Image.open(img_file).convert("L")
    else:
        img = Image.open(img_file).convert("L")
    w, h = img.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("L", (target_size, target_size), 255)
    offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
    canvas.paste(img, offset)
    return np.array(canvas).astype(np.float32)

# -------------------------
# Geometry feature extraction (same as before)
# -------------------------
def geometry_features_from_image(arr: np.ndarray) -> Dict[str, Any]:
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1.0)
    norm = mag / (mag.max() + 1e-12)

    v_sym = np.sum(np.abs(norm - np.flip(norm, axis=1)))
    v_score = float(max(0.0, 1 - (v_sym / norm.size)))

    h_sym = np.sum(np.abs(norm - np.flip(norm, axis=0)))
    h_score = float(max(0.0, 1 - (h_sym / norm.size)))

    center = (norm.shape[0] // 2, norm.shape[1] // 2)
    yy, xx = np.indices(norm.shape)
    angles = np.arctan2(yy - center[0], xx - center[1])
    radial_bins = np.linspace(-np.pi, np.pi, 32)
    hist, _ = np.histogram(angles, bins=radial_bins, weights=norm)
    lobe_count = int((hist > (0.3 * hist.max())).sum()) if hist.max() > 0 else 0

    fft1d = np.abs(np.fft.fft(hist))
    rot_harmonics = int((fft1d > (0.2 * fft1d.max())).sum()) if fft1d.max() > 0 else 0

    M = np.sum(norm) + 1e-12
    coords_y = np.sum(norm * np.indices(norm.shape)[0]) / M
    coords_x = np.sum(norm * np.indices(norm.shape)[1]) / M
    centroid_shift = float(np.sqrt((coords_x - center[1])**2 + (coords_y - center[0])**2) / (norm.shape[0] / 2.0))

    features = {
        "lobe_count": lobe_count,
        "v_score": round(v_score, 4),
        "h_score": round(h_score, 4),
        "rot_harmonics": rot_harmonics,
        "centroid_shift": round(centroid_shift, 4),
        "mag": mag  # for visualization if needed
    }
    return features

def geometry_code(features: Dict[str, Any]) -> str:
    return f"G{features['lobe_count']}-V{features['v_score']:.2f}-H{features['h_score']:.2f}-R{features['rot_harmonics']}"

# -------------------------
# Rule base (editable)
# -------------------------
DEFAULT_RULES = [
    {"id":"rule_container","label":"Container / Storage","conditions":{"v_score_min":0.6,"h_score_max":0.6,"lobe_max":2},"confidence":0.85,"explanation":"High vertical symmetry, low horizontal symmetry, few lobes → container"},
    {"id":"rule_flow","label":"Flow / Movement","conditions":{"h_score_min":0.5,"centroid_shift_min":0.05},"confidence":0.78,"explanation":"Horizontal symmetry + centroid shift → directional/flow"},
    {"id":"rule_duality","label":"Duality / Pairing","conditions":{"lobe_min":3,"rot_harmonics_min":2},"confidence":0.88,"explanation":"Multiple lobes & rotational harmonics → paired/ritual operator"},
    {"id":"rule_celestial","label":"Celestial / Cosmic","conditions":{"rot_harmonics_min":4,"lobe_min":4},"confidence":0.90,"explanation":"High rotational harmonic content & many lobes → cosmological"},
    {"id":"rule_numeric","label":"Numeric / Counting","conditions":{"lobe_max":1,"centroid_shift_max":0.02},"confidence":0.65,"explanation":"Centered, low-lobe marks → numeric"}
]

def rule_matches(features: Dict[str, Any], rule: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    c = rule.get("conditions", {})
    detail = {}
    # checks, store deltas for explain
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
    if "rot_harmonics_max" in c:
        detail["rot_harmonics"] = (features["rot_harmonics"], c["rot_harmonics_max"])
        if features["rot_harmonics"] > c["rot_harmonics_max"]:
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

# -------------------------
# Prototype fallback (for similarity)
# -------------------------
PROTOTYPES = {
    "Container / Storage": np.array([1.0, 0.8, 0.2, 0.5, 0.05]),
    "Flow / Movement": np.array([0.4, 0.3, 0.75, 0.8, 0.15]),
    "Duality / Pairing": np.array([0.8, 0.4, 0.4, 0.9, 0.05]),
    "Celestial / Cosmic": np.array([1.0, 0.5, 0.5, 1.0, 0.02]),
    "Numeric / Counting": np.array([0.1, 0.95, 0.95, 0.1, 0.01])
}

def fallback_predict(features: Dict[str, Any]) -> List[Tuple[str, float]]:
    fv = np.array([
        min(features["lobe_count"], 8) / 8.0,
        features["v_score"],
        features["h_score"],
        min(features["rot_harmonics"], 8) / 8.0,
        features["centroid_shift"]
    ])
    dists = {}
    for label, proto in PROTOTYPES.items():
        d = np.linalg.norm(fv - proto)
        dists[label] = float(d)
    maxd = max(dists.values()) if dists else 1.0
    sims = {k: 1 - (v / (maxd + 1e-12)) for k, v in dists.items()}
    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    return sorted_sims

# -------------------------
# k-NN learner (pure numpy)
# -------------------------
def build_knn_model(mappings: Dict[str, Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    # mappings: name -> {"saved_label": label, "features": features}
    X = []
    y = []
    for name, rec in mappings.items():
        f = rec["features"]
        vec = np.array([
            min(f["lobe_count"], 8) / 8.0,
            f["v_score"],
            f["h_score"],
            min(f["rot_harmonics"], 8) / 8.0,
            f["centroid_shift"]
        ])
        X.append(vec)
        y.append(rec["saved_label"])
    if len(X) == 0:
        return np.zeros((0,5)), []
    return np.vstack(X), y

def knn_predict_one(x, X_train, y_train, k=3):
    if X_train.shape[0] == 0:
        return None, 0.0
    # compute distances
    dists = np.linalg.norm(X_train - x.reshape(1,-1), axis=1)
    idx = np.argsort(dists)[:k]
    labels = [y_train[i] for i in idx]
    # vote
    from collections import Counter
    cnt = Counter(labels)
    label, count = cnt.most_common(1)[0]
    # confidence = fraction of k that agreed (simple)
    conf = count / float(k)
    return label, conf

# -------------------------
# Session state
# -------------------------
if "rule_base" not in st.session_state:
    st.session_state.rule_base = DEFAULT_RULES.copy()
if "user_mappings" not in st.session_state:
    st.session_state.user_mappings = {}  # name -> {"saved_label":..., "features":...}

# -------------------------
# UI: left column = upload & inference; right column = batch & tools
# -------------------------
col1, col2 = st.columns([2,1])

with col1:
    st.header("Single Symbol — Extract & Infer")
    uploaded = st.file_uploader("Upload one symbol (JPG/PNG) to analyze", type=["jpg","jpeg","png"])
    use_builtin = st.checkbox("Use uploaded workspace sheet (auto-split 3) if present", value=False)
    sample_sheet_path = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"

    if use_builtin and os.path.exists(sample_sheet_path):
        st.info(f"Using sheet at: {sample_sheet_path}")
        # allow user to pick which third
        which = st.selectbox("Pick slice from sheet (left/mid/right)", ["left","middle","right"])
        sheet = Image.open(sample_sheet_path).convert("L")
        w,h = sheet.size
        idx = {"left":0,"middle":1,"right":2}[which]
        left = int(idx * w / 3)
        right = int((idx+1) * w / 3)
        crop = sheet.crop((left,0,right,h)).resize((256,256))
        arr = np.array(crop).astype(np.float32)
        sym_name = f"sheet_{which}"
        features = geometry_features_from_image(arr)
    elif uploaded:
        try:
            arr = load_and_normalize(uploaded)
            sym_name = getattr(uploaded, "name", "uploaded_symbol")
            features = geometry_features_from_image(arr)
        except Exception as e:
            st.error(f"Could not load image: {e}")
            arr = None
            features = None
            sym_name = None
    else:
        arr = None
        features = None
        sym_name = None

    if arr is not None:
        st.subheader("Preview & FFT")
        st.image(safe_image(arr), caption="Normalized symbol (256×256)", use_column_width=True)
        mag = features["mag"]
        st.image(safe_image(mag), caption="FFT magnitude (log)", use_column_width=True)

        st.subheader("Extracted Features")
        st.json({k:v for k,v in features.items() if k!="mag"})

        # Geometry + rule inference
        code = geometry_code(features)
        st.write("Geometry code:", code)

        matches = []
        explain_details = []
        for r in st.session_state.rule_base:
            ok, detail = rule_matches(features, r)
            if ok:
                matches.append({"label": r["label"], "confidence": r["confidence"], "explanation": r["explanation"], "rule_id": r["id"], "detail": detail})
            else:
                # store negative detail for debugging
                explain_details.append({"rule_id": r["id"], "passed": False, "detail": detail})

        fallback = fallback_predict(features)

        st.subheader("Rule-based matches")
        if matches:
            for m in matches:
                st.write(f"- **{m['label']}** (rule confidence {m['confidence']:.2f})")
                st.caption(m['explanation'])
                if m.get("detail"):
                    st.write("Condition deltas (feature vs threshold):", m["detail"])
        else:
            st.write("_No explicit rule fired._")
            st.write("Rule checks (failure reasons):")
            for e in explain_details:
                st.write(f"- {e['rule_id']}: {e['detail']}")

        st.subheader("Fallback prototype similarities")
        for label, score in fallback[:5]:
            st.write(f"- {label} — similarity {score:.2f}")

        # Combined ranking
        combined = {}
        for label, score in fallback:
            combined[label] = 0.3 * score
        for m in matches:
            combined[m["label"]] = combined.get(m["label"], 0.0) + 0.7 * m["confidence"]
        ranked = sorted(combined.items(), key=lambda x: -x[1])
        st.subheader("Combined Hypotheses")
        for label, score in ranked:
            st.write(f"- **{label}** — combined score {score:.2f}")

        # Human-in-loop: save mapping
        st.subheader("Human-in-the-loop: Save mapping")
        choices = [lab for lab,_ in ranked] + [lab for lab,_ in fallback[:5] if lab not in [r[0] for r in ranked]]
        chosen_label = st.selectbox("Select label to save for this symbol", options=["(none)"] + choices)
        if st.button("Save mapping for this symbol"):
            if chosen_label and chosen_label != "(none)":
                st.session_state.user_mappings[sym_name] = {"saved_label": chosen_label, "features": features}
                st.success(f"Saved mapping: {sym_name} → {chosen_label}")
            else:
                st.info("Choose a label before saving.")

    # Show existing saved mappings
    if st.session_state.user_mappings:
        st.subheader("Saved mappings (session)")
        st.write("Count:", len(st.session_state.user_mappings))
        st.json(st.session_state.user_mappings)

    # k-NN training & predictions (live)
    st.markdown("---")
    st.subheader("Live k-NN Learner (from saved mappings)")
    X_train, y_train = build_knn_model(st.session_state.user_mappings)
    if X_train.shape[0] >= 1:
        st.write(f"Training set size: {X_train.shape[0]}")
        knn_k = st.slider("k (neighbors)", 1, min(7, max(1, X_train.shape[0])), 3)
        if features:
            xq = np.array([
                min(features["lobe_count"],8)/8.0,
                features["v_score"],
                features["h_score"],
                min(features["rot_harmonics"],8)/8.0,
                features["centroid_shift"]
            ])
            pred_label, pred_conf = knn_predict_one(xq, X_train, y_train, k=knn_k)
            if pred_label:
                st.write(f"k-NN prediction: **{pred_label}** (confidence {pred_conf:.2f})")
            else:
                st.write("k-NN: no prediction")
    else:
        st.info("No saved mappings yet — save a few mappings to enable k-NN.")

with col2:
    st.header("Batch Tools & Exports")

    # Batch-run over a folder
    batch_folder = st.text_input("Batch folder (path)", value="sample_data")
    if st.button("Run batch inference on folder"):
        if not os.path.exists(batch_folder):
            st.error("Folder not found.")
        else:
            rows = []
            for fn in sorted(os.listdir(batch_folder)):
                if fn.lower().endswith((".jpg",".jpeg",".png")):
                    path = os.path.join(batch_folder, fn)
                    try:
                        arr = load_and_normalize(path)
                        feats = geometry_features_from_image(arr)
                        # run rule-based + fallback combined
                        matches = []
                        for r in st.session_state.rule_base:
                            ok, _ = rule_matches(feats, r)
                            if ok:
                                matches.append((r["label"], r["confidence"]))
                        fallback = fallback_predict(feats)
                        # combined
                        combined = {}
                        for label, score in fallback:
                            combined[label] = 0.3 * score
                        for lab, conf in matches:
                            combined[lab] = combined.get(lab, 0.0) + 0.7 * conf
                        ranked = sorted(combined.items(), key=lambda x: -x[1])
                        top_label = ranked[0][0] if ranked else ""
                        top_score = ranked[0][1] if ranked else 0.0
                        rows.append({
                            "filename": fn,
                            "geometry_code": geometry_code(feats),
                            "top_label": top_label,
                            "top_score": round(top_score,3),
                            "features": feats
                        })
                    except Exception as e:
                        rows.append({"filename": fn, "error": str(e)})
            # Write CSV
            out_csv = "batch_inference_results.csv"
            with open(out_csv, "w", newline='') as f:
                w = csv.writer(f)
                w.writerow(["filename","geometry_code","top_label","top_score"])
                for r in rows:
                    if "error" in r:
                        w.writerow([r["filename"], "ERROR", r["error"], ""])
                    else:
                        w.writerow([r["filename"], r["geometry_code"], r["top_label"], r["top_score"]])
            st.success(f"Batch complete. {len(rows)} files processed. CSV: {out_csv}")
            st.write("Preview:")
            st.dataframe(rows[:20])

    # Auto-load from the workspace sheet (developer-specified path)
    st.markdown("---")
    st.write("Auto-load & split the workspace sheet (if you uploaded it earlier)")
    workspace_sheet_path = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"
    st.write(f"Workspace path used: `{workspace_sheet_path}`")
    if st.button("Auto-split sheet into 3 and infer"):
        if not os.path.exists(workspace_sheet_path):
            st.error("Sheet not found at workspace path.")
        else:
            sheet = Image.open(workspace_sheet_path).convert("L")
            w,h = sheet.size
            results = []
            for i,name in enumerate(["sheet_left","sheet_center","sheet_right"]):
                left = int(i * w / 3)
                right = int((i+1) * w / 3)
                crop = sheet.crop((left,0,right,h)).resize((256,256))
                arr = np.array(crop).astype(np.float32)
                feats = geometry_features_from_image(arr)
                fallback = fallback_predict(feats)
                matches = [r["label"] for r in st.session_state.rule_base if rule_matches(feats, r)[0]]
                combined = {}
                for label, score in fallback:
                    combined[label] = 0.3 * score
                for lab in matches:
                    # find confidence from rules
                    conf = next((r["confidence"] for r in st.session_state.rule_base if r["label"]==lab), 0.6)
                    combined[lab] = combined.get(lab,0.0) + 0.7 * conf
                ranked = sorted(combined.items(), key=lambda x: -x[1])
                top_label = ranked[0][0] if ranked else ""
                top_score = ranked[0][1] if ranked else 0.0
                results.append({"slice":name,"geometry_code":geometry_code(feats),"top_label":top_label,"top_score":round(top_score,3),"features":feats})
            # show results & allow export
            st.write("Auto-split results:")
            st.dataframe(results)
            out_path = "sheet_split_inference.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            st.success(f"Saved sheet split inference to {out_path}")

    st.markdown("---")
    st.subheader("Export / Import session mappings")
    if st.button("Export mappings (JSON)"):
        out = "symbol_mappings_session.json"
        with open(out, "w") as f:
            json.dump(st.session_state.user_mappings, f, indent=2)
        st.success(f"Exported to {out}")

    upload_json = st.file_uploader("Import mappings JSON", type=["json"])
    if upload_json:
        try:
            loaded = json.load(upload_json)
            # validate basic structure
            for k,v in loaded.items():
                if "saved_label" in v and "features" in v:
                    st.session_state.user_mappings[k] = v
            st.success("Imported mappings into session.")
        except Exception as e:
            st.error(f"Failed to import: {e}")

st.markdown("---")
st.caption("Notes: k-NN uses the 5-d geometric feature vector. Rule editor available earlier in the standalone app; rules are editable in the code (or we can add a rule-editor UI next).")
