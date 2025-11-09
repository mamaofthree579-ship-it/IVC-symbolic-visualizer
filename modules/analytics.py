# modules/analytics.py
import numpy as np
import pandas as pd

def generate_sample_data(n=6):
    """Generate a reproducible sample symbolic data matrix (n x n)."""
    np.random.seed(42)
    symbols = [f"Symbol_{i}" for i in range(n)]
    values = np.random.rand(n, n) * 1.0
    df = pd.DataFrame(values, index=symbols, columns=symbols)
    return df


def compute_resonance_matrix(df):
    """
    Compute a symmetric resonance matrix from a DataFrame.
    Uses row-wise cosine-like normalization to compute pairwise similarity.
    Returns a pandas DataFrame (NxN) with same index/columns as input.
    """
    mat = df.to_numpy(dtype=float)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = mat / norms
    resonance = np.dot(normalized, normalized.T)
    symbols = list(df.index)
    return pd.DataFrame(resonance, index=symbols, columns=symbols)


def find_resonant_clusters(matrix, threshold=0.8):
    """
    Simple threshold-based cluster finder: groups indices that share strong pairwise resonance.
    Returns list of sets (symbol names).
    """
    if isinstance(matrix, pd.DataFrame):
        labels = list(matrix.index)
        M = matrix.to_numpy()
    else:
        raise ValueError("matrix must be a pandas DataFrame")

    n = M.shape[0]
    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        group = {i}
        for j in range(n):
            if i != j and M[i, j] >= threshold:
                group.add(j)
        visited |= group
        clusters.append({labels[k] for k in sorted(group)})
    return clusters


def compute_symbol_energy(df):
    """
    Aggregate energy per symbol - here the mean resonance magnitude across its row.
    Accepts DataFrame (square) and returns a numpy array (length n).
    """
    if isinstance(df, pd.DataFrame):
        m = df.to_numpy()
    else:
        m = np.asarray(df)
    energy = np.nanmean(m, axis=1)
    return energy


def compute_energy_flow(df):
    """
    Compute a simple per-symbol flow vector based on gradients of the data.
    Returns an (n,3) numpy array where each row is (u,v,w) vector for that symbol.
    """
    if isinstance(df, pd.DataFrame):
        arr = df.to_numpy()
    else:
        arr = np.asarray(df, dtype=float)

    # Use gradients in 2D + a small z component from row differences
    gx = np.gradient(arr, axis=1).mean(axis=1)
    gy = np.gradient(arr, axis=0).mean(axis=1)
    gz = np.gradient(gx + gy)
    vectors = np.vstack([gx, gy, gz]).T

    # normalize per-row
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors


def evolve_matrix_step(base_matrix, step_index, amplitude=0.1):
    """
    Create a gentle evolution of the base resonance matrix for animation.
    Adds a small oscillatory perturbation that preserves symmetry.
    """
    if not isinstance(base_matrix, pd.DataFrame):
        raise ValueError("base_matrix must be a pandas DataFrame")

    n = base_matrix.shape[0]
    rng = np.random.default_rng(100 + (step_index % 1000))
    phase = np.sin(2.0 * np.pi * (step_index / 10.0))
    perturb = amplitude * phase * rng.standard_normal((n, n))

    # Make symmetric so resonance stays balanced
    perturb = (perturb + perturb.T) * 0.5

    mat = base_matrix.to_numpy() + perturb
    # Clip to [0,1] for visualization stability
    mat = np.clip(mat, 0.0, 1.0)

    return pd.DataFrame(mat, index=base_matrix.index, columns=base_matrix.columns)
