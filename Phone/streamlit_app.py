# app.py — Indus Resonance System + Meaning Inference Engine
import streamlit as st
import numpy as np
from PIL import Image
import os
import json

st.set_page_config(page_title="Indus Resonance + Meaning Engine", layout="centered")
st.title("Indus Symbol Resonance • Meaning Inference Engine")

# -------------------------
# SAFE NORMALIZER
# -------------------------
def safe_image(arr):
    arr = np.nan_to_num(arr)
    arr = arr - arr.min()
    maxv = arr.max()
    if maxv > 0:
        arr = arr / maxv
    return arr

# -------------------------
# UNIVERSAL LOADER (256x256)
# -------------------------
def load_and_normalize(img_file, target_size=256):
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
# GEOMETRY EXTRACTION
# -------------------------
def geometry_features_from_image(arr):
    # arr: normalized grayscale 0..255 (float)
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    mag = np.log(np.abs(fft_shift) + 1.0)
    norm = mag / mag.max()

    # vertical/horizontal symmetry scores (0..1, higher = more symmetric)
    v_sym = np.sum(np.abs(norm - np.flip(norm, axis=1)))
    v_score = float(max(0.0, 1 - (v_sym / norm.size)))

    h_sym = np.sum(np.abs(norm - np.flip(norm, axis=0)))
    h_score = float(max(0.0, 1 - (h_sym / norm.size)))

    # radial lobes (coarse angular histogram)
    center = (norm.shape[0] // 2, norm.shape[1] // 2)
    yy, xx = np.indices(norm.shape)
    angles = np.arctan2(yy - center[0], xx - center[1])
    radial_bins = np.linspace(-np.pi, np.pi, 32)
    hist, _ = np.histogram(angles, bins=radial_bins, weights=norm)
    lobe_count = int((hist > (0.3 * hist.max())).sum())

    # rotational harmonics (count of peaks in fft of angular histogram)
    fft1d = np.abs(np.fft.fft(hist))
    rot_harmonics = int((fft1d > (0.2 * fft1d.max())).sum())

    # centroid shift (how off-center the energy is)
    M = np.sum(norm)
    coords_y = np.sum(norm * np.indices(norm.shape)[0]) / (M + 1e-12)
    coords_x = np.sum(norm * np.indices(norm.shape)[1]) / (M + 1e-12)
    centroid_shift = float(np.sqrt((coords_x - center[1])**2 + (coords_y - center[0])**2) / (norm.shape[0] / 2.0))

    features = {
        "lobe_count": lobe_count,
        "v_score": round(v_score, 3),
        "h_score": round(h_score, 3),
        "rot_harmonics": rot_harmonics,
        "centroid_shift": round(centroid_shift, 3),
        "raw_mag": mag  # keep for visualization if needed
    }
    return features

def geometry_code(features):
    return f"G{features['lobe_count']}-V{features['v_score']:.2f}-H{features['h_score']:.2f}-R{features['rot_harmonics']}"

# -------------------------
# MEANING INFERENCE ENGINE (RULE-BASED + fallback)
# -------------------------
# Editable rule base: human-readable rules mapping geometric conditions -> semantic tags
DEFAULT_RULES = [
    {
        "id": "rule_container",
        "label": "Container / Storage",
        "conditions": {"v_score_min": 0.6, "h_score_max": 0.6, "lobe_max": 2},
        "confidence": 0.85,
        "explanation": "High vertical symmetry, low horizontal symmetry, few lobes → container-like shape"
    },
    {
        "id": "rule_flow",
        "label": "Flow / Movement / Transit",
        "conditions": {"h_score_min": 0.5, "centroid_shift_min": 0.05},
        "confidence": 0.78,
        "explanation": "Pronounced horizontal symmetry and centroid shift → directional/flow glyph"
    },
    {
        "id": "rule_duality",
        "label": "Duality / Pairing / Ritual Amplifier",
        "conditions": {"lobe_min": 3, "rot_harmonics_min": 2},
        "confidence": 0.88,
        "explanation": "Multiple lobes & rotational harmonics → paired or ritual-composite glyph"
    },
    {
        "id": "rule_celestial",
        "label": "Celestial / Cosmic Marker",
        "conditions": {"rot_harmonics_min": 4, "lobe_min": 4},
        "confidence": 0.90,
        "explanation": "High rotational harmonic content and many lobes → cosmological symbol"
    },
    {
        "id": "rule_numeric",
        "label": "Numeric / Counting Marker",
        "conditions": {"lobe_max": 1, "centroid_shift_max": 0.02},
        "confidence": 0.65,
        "explanation": "Very simple single-lobe centered marks → possible numeric/stroke markers"
    }
]

# Helper to evaluate a single rule
def rule_matches(features, rule):
    c = rule.get("conditions", {})
    # lower/upper checks; missing keys are ignored
    if "v_score_min" in c and features["v_score"] < c["v_score_min"]:
        return False
    if "v_score_max" in c and features["v_score"] > c["v_score_max"]:
        return False
    if "h_score_min" in c and features["h_score"] < c["h_score_min"]:
        return False
    if "h_score_max" in c and features["h_score"] > c["h_score_max"]:
        return False
    if "lobe_min" in c and features["lobe_count"] < c["lobe_min"]:
        return False
    if "lobe_max" in c and features["lobe_count"] > c["lobe_max"]:
        return False
    if "rot_harmonics_min" in c and features["rot_harmonics"] < c["rot_harmonics_min"]:
        return False
    if "rot_harmonics_max" in c and features["rot_harmonics"] > c["rot_harmonics_max"]:
        return False
    if "centroid_shift_min" in c and features["centroid_shift"] < c["centroid_shift_min"]:
        return False
    if "centroid_shift_max" in c and features["centroid_shift"] > c["centroid_shift_max"]:
        return False
    return True

# Fallback inference (simple distance to prototype vectors)
# Prototype vectors for semantic classes (tunable / extendable)
PROTOTYPES = {
    "Container / Storage": np.array([1.0, 0.8, 0.2, 0.5, 0.05]),  # [lobe_norm, v, h, rot, centroid]
    "Flow / Movement / Transit": np.array([0.4, 0.3, 0.75, 0.8, 0.15]),
    "Duality / Pairing": np.array([0.8, 0.4, 0.4, 0.9, 0.05]),
    "Celestial / Cosmic": np.array([1.0, 0.5, 0.5, 1.0, 0.02]),
    "Numeric / Counting": np.array([0.1, 0.95, 0.95, 0.1, 0.01])
}

def fallback_predict(features):
    # normalize features into comparable vector
    # lobe_count normalized by an assumed max (8), rot_harmonics normalized by 8, centroid shift already 0..~1
    fv = np.array([
        min(features["lobe_count"], 8) / 8.0,
        features["v_score"],
        features["h_score"],
        min(features["rot_harmonics"], 8) / 8.0,
        features["centroid_shift"]
    ])
    # compute distances
    dists = {}
    for label, proto in PROTOTYPES.items():
        d = np.linalg.norm(fv - proto)
        dists[label] = float(d)
    # convert to similarity scores
    maxd = max(dists.values())
    sims = {k: 1 - (v / (maxd + 1e-12)) for k, v in dists.items()}
    # return sorted by similarity
    sorted_sims = sorted(sims.items(), key=lambda x: -x[1])
    return sorted_sims

# -------------------------
# Persisted rules (in-session)
# -------------------------
if "rule_base" not in st.session_state:
    st.session_state.rule_base = DEFAULT_RULES.copy()

if "user_mappings" not in st.session_state:
    st.session_state.user_mappings = {}  # symbol_name -> chosen_label

# -------------------------
# UI — upload or built-in
# -------------------------
st.sidebar.header("Auto-sample (optional)")
SAMPLE_SHEET = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"
if os.path.exists(SAMPLE_SHEET):
    if st.sidebar.button("Auto-load sample sheet (split 3)"):
        # split vertical thirds if present
        sheet = Image.open(SAMPLE_SHEET).convert("L")
        w, h = sheet.size
        for i, name in enumerate(["sample_A", "sample_B", "sample_C"]):
            left = int(i * w / 3)
            right = int((i + 1) * w / 3)
            crop = sheet.crop((left, 0, right, h)).resize((256,256))
            arr = np.array(crop).astype(np.float32)
            st.session_state.user_mappings[name] = {"saved_label": None, "features": geometry_features_from_image(arr)}
        st.success("Sample sheet split and loaded into session mappings (names: sample_A/B/C).")

st.header("Upload or choose built-in symbol")
uploaded = st.file_uploader("Upload a symbol (JPG/PNG) — single file", type=["jpg","jpeg","png"])
builtin_choice = None
if st.checkbox("Use built-in example (if available)"):
    # look for any of the common filenames in CWD
    for f in ["jar.jpg", "fish.jpg", "double_fish.jpg", "jar.png", "fish.png"]:
        if os.path.exists(f):
            builtin_choice = f
            break
    if builtin_choice:
        st.info(f"Using built-in file: {builtin_choice}")
        uploaded = builtin_choice

# load image and compute features
sym_name = None
features = None
if uploaded:
    try:
        if isinstance(uploaded, str):
            arr = load_and_normalize(uploaded)
            sym_name = os.path.basename(uploaded)
        else:
            arr = load_and_normalize(uploaded)
            sym_name = getattr(uploaded, "name", "uploaded_symbol")
        features = geometry_features_from_image(arr)
    except Exception as e:
        st.error(f"Could not load image: {e}")

# -------------------------
# Display features, geometry code, inference
# -------------------------
if features:
    st.subheader("Extracted Geometry Features")
    st.json(features)

    code = geometry_code(features)
    st.write("Geometry code:", code)

    # Run rule-based inference
    matches = []
    for r in st.session_state.rule_base:
        if rule_matches(features, r):
            matches.append({"label": r["label"], "confidence": r["confidence"], "explanation": r["explanation"], "rule_id": r["id"]})

    # fallback predictions
    fallback = fallback_predict(features)  # list of (label, score)

    # Present results
    st.subheader("Meaning Inference — Rule-based (explicit)")
    if matches:
        for m in matches:
            st.write(f"- **{m['label']}** (confidence {m['confidence']:.2f}) — {m['explanation']}")
    else:
        st.write("_No explicit rule match._")

    st.subheader("Meaning Inference — Fallback (similarity to prototypes)")
    for label, score in fallback[:5]:
        st.write(f"- {label} — similarity {score:.2f}")

    # Combined ranked view
    st.subheader("Combined Hypotheses (ranked)")
    # Build a combined score: rule_confidence * 0.7 + fallback_similarity * 0.3 (if rule matched)
    combined = {}
    for label, score in fallback:
        combined[label] = 0.3 * score  # base from fallback
    for m in matches:
        combined[m["label"]] = combined.get(m["label"], 0.0) + 0.7 * m["confidence"]

    ranked = sorted(combined.items(), key=lambda x: -x[1])
    for label, score in ranked:
        st.write(f"- **{label}** — combined score {score:.2f}")

    # Allow user to pick & save mapping
    st.subheader("Human-in-the-loop: select label / save mapping")
    choices = [lab for lab, _ in ranked] + [lab for lab, _ in fallback[:5] if lab not in [r[0] for r in ranked]]
    chosen_label = st.selectbox("Choose semantic label for this symbol (or leave blank)", options=["(none)"] + choices)
    if st.button("Save mapping for this symbol"):
        if chosen_label and chosen_label != "(none)":
            st.session_state.user_mappings[sym_name] = {"saved_label": chosen_label, "features": features}
            st.success(f"Saved mapping: {sym_name} → {chosen_label}")
        else:
            st.info("Choose a label before saving.")

    # Show saved mappings
    if st.session_state.user_mappings:
        st.subheader("Saved mappings (session)")
        st.json(st.session_state.user_mappings)

# -------------------------
# RULE EDITOR (simple)
# -------------------------
st.sidebar.header("Rule Editor")
if st.sidebar.button("Show current rule base"):
    st.sidebar.write(json.dumps(st.session_state.rule_base, indent=2))

st.sidebar.markdown("Add a simple rule (human-readable conditions):")
new_label = st.sidebar.text_input("Rule label", "")
col1, col2 = st.sidebar.columns(2)
vmin = col1.number_input("v_score_min", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
hmax = col2.number_input("h_score_max", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
lmax = st.sidebar.number_input("lobe_max (int)", min_value=0, max_value=20, value=10)

if st.sidebar.button("Add rule"):
    if new_label.strip():
        new_rule = {"id": f"r_{len(st.session_state.rule_base)+1}", "label": new_label,
                    "conditions": {"v_score_min": float(vmin), "h_score_max": float(hmax), "lobe_max": int(lmax)},
                    "confidence": 0.6, "explanation": "User-added rule"}
        st.session_state.rule_base.append(new_rule)
        st.sidebar.success(f"Added rule: {new_label}")
    else:
        st.sidebar.error("Provide a rule label.")

# -------------------------
# Export saved mappings
# -------------------------
st.sidebar.header("Export")
if st.sidebar.button("Export mappings to JSON"):
    out_path = "symbol_mappings_session.json"
    with open(out_path, "w") as f:
        json.dump(st.session_state.user_mappings, f, indent=2)
    st.sidebar.success(f"Exported to {out_path}")
    st.sidebar.write(f"Path: {out_path}")

st.sidebar.caption("Meaning engine uses explicit rules + prototype fallback. Edit rules to refine behavior.")
