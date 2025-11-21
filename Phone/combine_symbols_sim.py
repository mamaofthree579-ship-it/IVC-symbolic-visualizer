"""
combine_symbols_sim.py

Simulate multi-symbol optical (Fresnel) and acoustic (frequency-domain phasor)
combination by mathematically summing complex fields / phasors.

Dependencies: numpy, pillow, matplotlib, optionally scipy for gaussian smoothing.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------- Utilities ----------
def load_image_as_heightmap(path, size=512, invert=False):
    """
    Load image (PNG/JPG) and return a normalized heightmap (float array 0..1).
    Black pixels -> 1.0 (raised) by default; white -> 0.0.
    """
    img = Image.open(path).convert('L')
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)
    # Normalize: black (0) -> 1, white (255) -> 0
    height = 1.0 - (arr / 255.0)
    if invert:
        height = 1.0 - height
    return height

def spatial_fft(height):
    H = np.fft.fftshift(np.fft.fft2(height))
    mag = np.abs(H)
    return np.log1p(mag)

# ---------- Fresnel propagation (scalar approx) ----------
def fresnel_propagation_from_phase(height, wavelength=532e-9, z=0.15, dx=10e-6):
    """
    Treat height (0..1) as a normalized phase profile.
    Returns propagated complex field U2 on observation plane.
    dx = sampling spacing in meters (physical size = dx * N)
    """
    # Convert normalized height to phase (scale factor alpha to tune strength)
    alpha = 2*np.pi  # scale factor: converts normalized height -> phase radians (tunable)
    field = np.exp(1j * alpha * height)  # complex pupil field

    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dx)
    FX, FY = np.meshgrid(np.fft.fftshift(fx), np.fft.fftshift(fy))

    Hf = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2) * (2*np.pi)**2)
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))
    U2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F * Hf)))
    return U2

# ---------- Combine complex fields with phase/weight offsets ----------
def combine_fields(fields_complex, weights=None, phase_offsets_deg=None):
    """
    fields_complex: list of complex arrays (same shape)
    weights: list of scalars (amplitude multipliers)
    phase_offsets_deg: list of degrees to phase-shift each field before summing
    Returns combined complex field and intensity.
    """
    n = len(fields_complex)
    if weights is None:
        weights = [1.0]*n
    if phase_offsets_deg is None:
        phase_offsets_deg = [0.0]*n

    combined = np.zeros_like(fields_complex[0], dtype=np.complex128)
    for fld, w, ph in zip(fields_complex, weights, phase_offsets_deg):
        phase_rad = np.deg2rad(ph)
        combined += w * fld * np.exp(1j * phase_rad)
    intensity = np.abs(combined)**2
    return combined, intensity

# ---------- Acoustic frequency-domain combiner ----------
def combine_acoustic_spectra(spectra_list, freqs, phase_offsets_deg=None, weights=None):
    """
    spectra_list: list of amplitude arrays (same length) -> amplitude vs freq for each symbol
    freqs: frequency axis (Hz) corresponding to amplitude arrays
    phase_offsets_deg: list of phase offsets per symbol (deg) at each freq (single value per symbol currently)
    weights: amplitude weights per symbol
    Returns complex combined spectrum (complex-valued) and magnitude.
    """
    n = len(spectra_list)
    if weights is None:
        weights = [1.0]*n
    if phase_offsets_deg is None:
        phase_offsets_deg = [0.0]*n

    # Build complex phasors and sum
    combined = np.zeros_like(spectra_list[0], dtype=np.complex128)
    for spec, w, ph in zip(spectra_list, weights, phase_offsets_deg):
        phi = np.deg2rad(ph)
        combined += w * spec * np.exp(1j * phi)
    return combined, np.abs(combined)

# ---------- IO / plotting ----------
def show_and_save(img, title, fname=None, cmap='magma'):
    plt.figure(figsize=(5,5))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    if fname:
        plt.savefig(fname, bbox_inches='tight', dpi=200)
    plt.show()

# ---------- Example workflow ----------
if __name__ == "__main__":
    # Example symbol images — replace with your own paths
    # The developer-provided uploaded file path (example)
    example_path = "/mnt/data/A_digital_vector_image_displays_three_black_Indus_.png"

    # For multi-symbol test, supply a list of file paths
    # If you only have one file with multiple symbols (like the three-icons image),
    # you may first crop/extract regions into separate files and give their paths here.
    symbol_paths = [
        example_path,         # symbol A (use actual jar.png in practice)
        example_path,         # symbol B (placeholder; replace)
        example_path          # symbol C (placeholder; replace)
    ]

    # Load each as heightmap (same size)
    heightmaps = [load_image_as_heightmap(p, size=512) for p in symbol_paths]

    # Compute complex propagated fields for each symbol
    wavelength = 532e-9  # green light (tune as needed)
    z = 0.20              # propagation distance (m)
    dx = 10e-6            # sample spacing (m) - tune if using different physical scales

    fields = []
    for hm in heightmaps:
        U2 = fresnel_propagation_from_phase(hm, wavelength=wavelength, z=z, dx=dx)
        fields.append(U2)

    # Example: weights and phase offsets (deg) for each symbol activation
    weights = [1.0, 0.8, 1.2]
    phase_offsets_deg = [0.0, 45.0, 90.0]

    # Combine fields and compute intensity
    combined_complex, combined_intensity = combine_fields(fields, weights=weights, phase_offsets_deg=phase_offsets_deg)

    # Show/save individual intensities and combined
    for i, U in enumerate(fields):
        I = np.abs(U)**2
        show_and_save(I / I.max(), f"Symbol {i+1} Intensity (normalized)", fname=f"symbol_{i+1}_intensity.png")

    show_and_save(combined_intensity / combined_intensity.max(), "Combined Intensity (normalized)", fname="combined_intensity.png")

    # ---------- Acoustic example ----------
    # Suppose you measured single-symbol amplitude spectra (toy example below)
    freqs = np.linspace(1, 1000, 1000)  # 1..1000 Hz
    # Toy spectral shapes (peaks at different freqs)
    spec1 = np.exp(-0.5*((freqs-28)/8)**2)      # jar-like low peak
    spec2 = 0.7 * np.exp(-0.5*((freqs-48)/6)**2) # fish-like
    spec3 = 0.9 * np.exp(-0.5*((freqs-88)/4)**2) # double-fish-like

    spectra_list = [spec1, spec2, spec3]
    acoust_phase_offsets = [0.0, 45.0, 90.0]  # example degrees
    acoust_weights = [1.0, 0.9, 1.1]

    combined_spec_complex, combined_spec_mag = combine_acoustic_spectra(spectra_list, freqs, phase_offsets_deg=acoust_phase_offsets, weights=acoust_weights)

    # Plot combined acoustic magnitude
    plt.figure(figsize=(8,3))
    plt.plot(freqs, spec1, label='spec1 (jar)')
    plt.plot(freqs, spec2, label='spec2 (fish)')
    plt.plot(freqs, spec3, label='spec3 (double-fish)')
    plt.plot(freqs, combined_spec_mag, label='combined (mag)', linewidth=2, color='k')
    plt.xlim(0,200)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (arb.)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("combined_acoustic_spectrum.png", dpi=200)
    plt.show()

    # Save combined complex data for further processing
    np.save("combined_complex_field.npy", combined_complex)
    np.save("combined_intensity.npy", combined_intensity)
    np.save("combined_spec_complex.npy", combined_spec_complex)
    np.save("combined_spec_mag.npy", combined_spec_mag)

    print("Done. Saved combined_intensity.png, combined_acoustic_spectrum.png, and .npy outputs.")
