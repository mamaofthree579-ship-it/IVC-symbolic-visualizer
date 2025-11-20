
# Signal Test Protocol — Indus Waveform Project (Non-destructive / Replica-first)

## Overview
This protocol specifies step-by-step procedures to test whether Indus glyph shapes and seal objects exhibit symbol-specific resonance, diffraction, or EM responses under controlled laboratory conditions. All tests MUST begin with high-fidelity replicas unless explicit permission is granted by museums for non-invasive testing of originals.

## Safety & Permissions
- Obtain written permission for any tests on original artifacts. Prioritize non-destructive and non-contact methods.
- Use replicas for destructive or intensive testing.
- Laser safety: use appropriate goggles and interlocks.
- RF testing: ensure compliance with local transmission licenses for conducted emissions.

## Equipment (minimum kit)
- Laser Doppler Vibrometer (LDV) with scanning head
- Function generator and amplifier (audio to ultrasonic range)
- Miniature shaker and contact transducers
- Accelerometers (micro-g)
- Network analyzer + near-field RF probes
- Coherent laser and CCD/CMOS camera for diffraction imaging
- 3D scanner / structured-light or micro-CT
- 3D printer (SLA) and ceramic kiln for replicas
- Data acquisition system (sample rates ≥ 96 kHz for acoustic; appropriate for optical/RF)

## Replica preparation
1. Scan object using structured-light or micro-CT (resolution ≤ 50 μm).
2. Produce SLA resin print; smooth to match surface finish. Create fired steatite-like replica per Replicas_Manufacturing.md.
3. Label each replica with unique ID; record all fabrication parameters.

## Baseline characterization (all replicas)
- Measure dimensions (mm), mass (g), and center of mass.
- 3D height map export (grayscale depth image) at consistent resolution (e.g., 2048×2048 for small seals).
- Optical reflectance spectra (400–1000 nm) for surface finish notes.
- Material permittivity measurement (if possible).

## Acoustic sweep (20 Hz–20 kHz)
1. Mount replica on vibration-isolated stage, or test free-free configuration supported at nodal points (document).
2. Use function generator to run logarithmic sweep from 20 Hz to 20 kHz at low amplitude.
3. Record surface velocity using LDV with scanning grid (suggest 512–1024 sample points across object).
4. Repeat sweep with contact miniature shaker attached at standard location.
5. Extract peak frequencies and mode shapes via FFT and modal analysis.

## Extended acoustic/ultrasound (20 kHz–1 MHz)
1. Use contact ultrasound transducers for high-frequency sweeps.
2. Use scanning LDV for mapping; sample rates must exceed Nyquist for highest freq.

## Optical diffraction (visible/near-IR)
1. Mount coherent laser (λ = 532 nm or 633 nm) and illuminate relief at controlled distances (near-field and far-field).
2. Capture intensity patterns on screen and CCD at multiple distances and angles.
3. Compare measured intensity with scalar Fresnel diffraction simulation using the 3D height map as a phase mask.

## RF / Electromagnetic testing (MHz–GHz)
1. Use vector network analyzer (VNA) with near-field probe to sweep 1 MHz–6 GHz (or broader depending on equipment).
2. Place sample on low-loss dielectric support; map reflected and transmitted S-parameters.
3. For high frequency/THz behaviour, collaborate with specialized facilities.

## Multi-symbol composite tests
1. Place multiple glyph replicas in sequences matching archaeological arrangements.
2. Drive acoustic sources with controlled phase offsets; illuminate with multiple laser sources with programmable phase where possible (SLM recommended).
3. Measure interference patterns in mechanical (LDV) and optical (CCD) domains.

## Data Logging & Formats
- Save raw LDV time-series as .wav (floating point) with metadata JSON sidecar.
- Save 3D scans as .ply or .stl; height maps as 16-bit PNG TIFF.
- Store S-parameter data in .s1p/.s2p files.
- Use `Signal_Results.csv` schema for results logging (columns: replica_id, sign_id, test_type, freq_start, freq_end, peak_freq_Hz, Q_factor, amplitude_dB, phase_deg, LDV_map_ref, notes).

## Replication & Controls
- For each sign, test at least 3 independently manufactured replicas (different materials: resin, fired steatite, metal-backed).
- Include control objects: blank steatite disk of similar mass/size; randomized engraving to compare to patterned glyphs.

## Analysis
- Perform modal FEM simulations to compare measured peaks to predicted modes.
- Compute 2D spatial FFT of the height map to extract dominant spatial frequencies.
- Correlate experimental peak frequencies across sign families using clustering and mutual information metrics.

## Reporting
- Record negative results and publish them; absence of symbol-specific resonance is an important falsification outcome.
- Maintain open data repository with raw files and replication metadata.
