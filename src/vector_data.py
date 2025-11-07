import numpy as np

def sample_water_symbol(mode='default'):
    """Return sample vector field arrays and metadata.

    Returns: dict with keys: X, Y, U, V, magnitude, meta
    - X, Y: grid coordinates
    - U, V: vector components
    - magnitude: vector lengths
    - meta: metadata dict (name, description, suggested_frequency_hz)
    """
    # Create a circular, spiral-like flow pattern with a gentle wave component
    theta = np.linspace(0, 2 * np.pi, 36)
    r = np.linspace(0.2, 1.8, 12)
    R, T = np.meshgrid(r, theta)
    # polar to cartesian coordinates for grid points
    X = (R * np.cos(T)).flatten()
    Y = (R * np.sin(T)).flatten()

    # base spiral vector field (rotational + radial)
    U = -0.6 * np.sin(T).flatten() + 0.05 * (X)
    V =  0.6 * np.cos(T).flatten() + 0.05 * (Y)

    # add a gentle wave modulation to simulate "water ripple" resonance
    wave = 0.15 * np.sin(3.0 * T).flatten()
    U = U + wave * np.cos(T).flatten()
    V = V + wave * np.sin(T).flatten()

    magnitude = np.sqrt(U**2 + V**2)

    if mode == 'calm':
        U *= 0.5
        V *= 0.5
    elif mode == 'intense':
        U *= 1.4
        V *= 1.4

    meta = {
        'name': 'sample_nibi_flow',
        'description': 'Neutralized sample vector field inspired by water glyph geometry. For research/visualization only.',
        'suggested_frequency_hz': [432, 528],
        'notes': 'Use respectfully. This is a simplified geometric model for visualization.'
    }

    return {'X': X, 'Y': Y, 'U': U, 'V': V, 'magnitude': magnitude, 'meta': meta}
