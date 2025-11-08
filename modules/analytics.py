import numpy as np
import pandas as pd

def generate_sample_data(n=5):
    """Generate sample symbolic data."""
    np.random.seed(42)
    symbols = [f"Symbol_{i}" for i in range(n)]
    values = np.random.rand(n, n)
    df = pd.DataFrame(values, columns=symbols, index=symbols)
    return df


def compute_resonance_matrix(df):
    """Compute cosine similarity as a symbolic resonance matrix."""
    data = df.to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    resonance = np.dot(normalized, normalized.T)
    return pd.DataFrame(resonance, index=df.index, columns=df.columns)


def find_resonant_clusters(matrix, threshold=0.8):
    """Identify clusters of symbols with high resonance."""
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


def compute_energy_flow(df):
    """
    Compute symbolic energy flow vectors between symbols.
    Modeled as directional gradients of resonance.
    """
    matrix = compute_resonance_matrix(df)
    n = len(matrix)
    symbols = matrix.index.tolist()

    values = matrix.to_numpy()
    grad_x = np.gradient(values, axis=0)
    grad_y = np.gradient(values, axis=1)

    flow_vectors = {}
    for i, sym in enumerate(symbols):
        gx = np.mean(grad_x[i])
        gy = np.mean(grad_y[i])
        gz = np.sin(gx * gy) * 0.5  # resonance depth
        norm = np.sqrt(gx**2 + gy**2 + gz**2) + 1e-9
        flow_vectors[sym] = (gx / norm, gy / norm, gz / norm)

    return flow_vectors


def compute_symbol_energy(df):
    """
    Compute total resonance energy per symbol.
    """
    matrix = compute_resonance_matrix(df)
    energy = matrix.sum(axis=1)
    return pd.DataFrame({"Symbol": matrix.index, "Energy": energy})
