import numpy as np
import pandas as pd

# --- Data Generation ---
def generate_sample_data(n=6):
    np.random.seed(42)
    symbols = [f"Symbol_{i}" for i in range(n)]
    values = np.random.rand(n, n)
    df = pd.DataFrame(values, columns=symbols, index=symbols)
    return df

# --- Resonance Matrix ---
def compute_resonance_matrix(df):
    data = df.to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    resonance = np.dot(normalized, normalized.T)
    return pd.DataFrame(resonance, index=df.index, columns=df.columns)

# --- Resonant Clusters ---
def find_resonant_clusters(matrix, threshold=0.8):
    clusters = []
    n = matrix.shape[0]
    visited = set()
    for i in range(n):
        if i in visited:
            continue
        cluster = {i}
        for j in range(n):
            if i != j and matrix.iloc[i, j] >= threshold:
                cluster.add(j)
        visited |= cluster
        clusters.append({matrix.index[k] for k in cluster})
    return clusters

# --- Symbolic Energy ---
def compute_symbol_energy(df):
    """Energy as normalized magnitude per symbol."""
    return np.linalg.norm(df.values, axis=1)

# --- Energy Flow Field ---
def compute_energy_flow(df):
    """Compute directional energy vectors based on gradients."""
    data = df.to_numpy()
    gx, gy = np.gradient(data)
    gz = np.gradient(gx + gy, axis=0)
    flat_flow = np.vstack([
        gx.mean(axis=1),
        gy.mean(axis=1),
        gz.mean(axis=1)
    ]).T
    return flat_flow
