# -----------------------------
# ADDITIONAL TOOLS: EXPORT / SWEEP / REPORT
# Paste this block near the bottom of your app where arrs, names, dna_list, features_list exist.
# Requires: numpy, Pillow (PIL), matplotlib, io, zipfile, base64
# -----------------------------
import csv, zipfile, base64, matplotlib.pyplot as plt, tempfile, time
from PIL import Image
import io, html

# Workspace sheet path (as requested)
WORKSPACE_SHEET_URL = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"

# ---------- 1) Export CSV pack ----------
def export_features_and_dna_zip(names, features_list, dna_list, zip_name="indus_export.zip"):
    # Create in-memory ZIP of CSVs
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # features CSV
        feats_csv = io.StringIO()
        writer = csv.writer(feats_csv)
        header = ["symbol"] + list(features_list[0].keys())
        writer.writerow(header)
        for nm, feats in zip(names, features_list):
            # features may include 'mag' — skip or stringify
            row = [nm] + [json.dumps(feats.get(k)) if isinstance(feats.get(k), (dict,list)) else feats.get(k) for k in features_list[0].keys()]
            writer.writerow(row)
        zf.writestr("features_summary.csv", feats_csv.getvalue())

        # DNA CSVs
        for nm, dna in zip(names, dna_list):
            dna_csv = io.StringIO()
            writer = csv.writer(dna_csv)
            writer.writerow(["idx","dna_value"])
            for i, val in enumerate(dna):
                writer.writerow([i, float(val)])
            safe_name = nm.replace(" ", "_")
            zf.writestr(f"{safe_name}_dna.csv", dna_csv.getvalue())

    mem.seek(0)
    return mem.getvalue()

# UI buttons for export
st.header("Export: DNA & Features")
if st.button("Export all DNA + features as ZIP"):
    if len(names) == 0:
        st.info("No symbols to export.")
    else:
        zip_bytes = export_features_and_dna_zip(names, features_list, dna_list)
        st.download_button("⬇ Download export ZIP", zip_bytes, file_name="indus_export.zip", mime="application/zip")

# ---------- 2) Live sweep demo (optical and acoustic) ----------
# utility: build a frame for optical interference sweep (phase)
def build_optical_frame(arrs_for_sweep, weights, phase_offsets_rad, alpha=4*np.pi):
    # produce combined intensity map as in ritual_optical_simulator, but return normalized image
    fields = [np.exp(1j * alpha * a) for a in arrs_for_sweep]
    combined = np.zeros_like(fields[0], dtype=np.complex128)
    for f,w,ph in zip(fields, weights, phase_offsets_rad):
        combined += w * f * np.exp(1j*ph)
    intensity = np.abs(np.fft.fftshift(np.fft.fft2(combined)))
    img = np.log(1 + intensity.real)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return (img * 255).astype('uint8')  # 0..255 uint8 frames

def make_optical_sweep_gif(arrs_for_sweep, weights, start_phase_deg=0, stop_phase_deg=360, frames=24, duration_s=2.0):
    # create frames sweeping phases from start to stop
    frames_list = []
    for t in range(frames):
        phase = np.deg2rad(start_phase_deg + (stop_phase_deg - start_phase_deg) * (t / float(frames-1)))
        frame = build_optical_frame(arrs_for_sweep, weights, [phase]*len(arrs_for_sweep))
        pil = Image.fromarray(frame).convert("L").resize((512,512))
        frames_list.append(pil)
    # save GIF in memory
    buf = io.BytesIO()
    frames_list[0].save(buf, format='GIF', save_all=True, append_images=frames_list[1:], duration=int(1000*duration_s/frames), loop=0)
    buf.seek(0)
    return buf.getvalue()

st.header("Live Sweep Demo")
if len(arrs) >= 2:
    start_deg = st.number_input("Start phase (deg)", value=0, min_value=0, max_value=360)
    end_deg = st.number_input("End phase (deg)", value=360, min_value=0, max_value=360)
    frames = st.slider("Frames", 8, 60, 24)
    duration = st.slider("GIF duration (s)", 1.0, 6.0, 2.0)
    chosen_indices = st.multiselect("Pick symbols (indices) to include in sweep", options=list(range(len(arrs))), default=list(range(min(2,len(arrs)))))
    if st.button("Generate sweep GIF"):
        if len(chosen_indices) < 1:
            st.info("Pick at least one symbol.")
        else:
            arrs_sel = [arrs[i] for i in chosen_indices]
            weights = [1.0]*len(arrs_sel)
            gif_bytes = make_optical_sweep_gif(arrs_sel, weights, start_deg, end_deg, frames=frames, duration_s=duration)
            st.image(gif_bytes)
            st.download_button("⬇ Download sweep GIF", gif_bytes, file_name="optical_sweep.gif", mime="image/gif")

# optional acoustic sweep: vary a base frequency mapped from DNA and show combined envelope frames
def make_acoustic_sweep_gif(dna_list_sel, freqlow, freqhigh, frames=24, duration_s=2.0):
    freqs = np.linspace(1,2000,2000)
    frames_list = []
    for t in range(frames):
        fbase = freqlow + (freqhigh - freqlow) * (t / float(frames-1))
        combined = np.zeros_like(freqs)
        for dna in dna_list_sel:
            # map dna centroid to harmonic stack centered near fbase
            centroid = np.sum(np.arange(len(dna)) * dna) / (dna.sum()+1e-12)
            base = fbase * (0.5 + centroid)  # slight variation
            # synth gaussian peaks
            for mult in [1,2,3]:
                peak = base*mult
                combined += np.exp(-0.5*((freqs - peak)/(peak*0.03+1e-6))**2)
        # normalize and plot as small line image
        im = np.zeros((128, 512), dtype=np.uint8)
        # scale combined envelope to 0..127
        env = combined[:800]
        env = env - env.min()
        if env.max() > 0:
            env = env / env.max()
        ys = (env * 127).astype(np.uint8)
        # draw line horizontally
        for x, y in enumerate(ys):
            im[127-y:, x] = 255
        pil = Image.fromarray(im).convert("L")
        frames_list.append(pil)
    buf = io.BytesIO()
    frames_list[0].save(buf, format='GIF', save_all=True, append_images=frames_list[1:], duration=int(1000*duration_s/frames), loop=0)
    buf.seek(0)
    return buf.getvalue()

st.subheader("Acoustic sweep (animated envelope)")
if st.button("Create acoustic sweep GIF (default 40→400 Hz)"):
    if len(dna_list) == 0:
        st.info("No DNA available.")
    else:
        gif_bytes = make_acoustic_sweep_gif(dna_list[:min(4, len(dna_list))], freqlow=40, freqhigh=400, frames=30, duration_s=3.0)
        st.image(gif_bytes)
        st.download_button("⬇ Download acoustic sweep GIF", gif_bytes, file_name="acoustic_sweep.gif", mime="image/gif")

# ---------- 3) Automated hypothesis report (HTML) ----------
def generate_html_report(names, features_list, dna_list, cosmic_list, include_images=True):
    # Build HTML with embedded base64 images of FFT and DNA charts
    html_parts = ["<html><head><meta charset='utf-8'><title>Indus Hypothesis Report</title></head><body>"]
    html_parts.append("<h1>Indus Symbol Hypothesis Report</h1>")
    html_parts.append(f"<p>Workspace sheet used (if any): <code>{html.escape(WORKSPACE_SHEET_URL)}</code></p>")
    for i, (nm, feats, dna, cosmic) in enumerate(zip(names, features_list, dna_list, cosmic_list)):
        html_parts.append(f"<h2>Symbol {i+1}: {html.escape(nm)}</h2>")
        html_parts.append("<h3>Key features</h3><ul>")
        for k, v in feats.items():
            if k == "mag": continue
            html_parts.append(f"<li><b>{html.escape(str(k))}</b>: {html.escape(str(v))}</li>")
        html_parts.append("</ul>")
        html_parts.append("<h3>Cosmic detector</h3>")
        html_parts.append(f"<p>circularity_score: {cosmic.get('circularity_score')}</p>")
        if cosmic.get("rings"):
            html_parts.append(f"<p>detected rings: {len(cosmic.get('rings'))}</p>")
        # generate small FFT PNG and embed as base64
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4,2))
        mag = np.array(feats["mag"])
        ax.imshow(mag, cmap="magma")
        ax.axis("off")
        buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        html_parts.append(f"<h3>FFT (visual)</h3><img src='data:image/png;base64,{b64}' style='max-width:600px;'>")
        # DNA plot
        fig2, ax2 = plt.subplots(figsize=(6,1.2))
        ax2.plot(dna, linewidth=1)
        ax2.set_title("DNA fingerprint (256)")
        ax2.axis("off")
        buf2 = io.BytesIO(); fig2.savefig(buf2, format="png", bbox_inches="tight"); plt.close(fig2)
        buf2.seek(0)
        b64_2 = base64.b64encode(buf2.read()).decode('ascii')
        html_parts.append(f"<h3>DNA</h3><img src='data:image/png;base64,{b64_2}' style='max-width:800px;'>")
    html_parts.append("</body></html>")
    return "\n".join(html_parts)

st.header("Automated Hypothesis Report")
if st.button("Generate HTML report (downloadable)"):
    report_html = generate_html_report(names, features_list, dna_list, cosmic_list)
    st.download_button("⬇ Download HTML report", report_html.encode('utf-8'), file_name="indus_hypothesis_report.html", mime="text/html")
    st.success("Report generated. Open the HTML in a browser and 'Save As → PDF' to create a PDF.")

# Quick utility: show the workspace sheet file path (as requested, local path provided)
st.markdown("**Workspace sheet path (local file URL):**")
st.code(WORKSPACE_SHEET_URL)
