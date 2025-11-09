import numpy as np
import pandas as pd

def generate_sample_data(n=6):
    np.random.seed(42)
    symbols = [f"Symbol_{i}" for i in range(n)]
    values = np.random.rand(n, n)
    return pd.DataFrame(values, columns=symbols, index=symbols)

def compute_resonance_matrix(df):
    data = df.to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    resonance = np.dot(normalized, normalized.T)
    return pd.DataFrame(resonance, index=df.index, columns=df.columns)

def find_resonant_clusters(matrix, threshold=0.8):
    clusters = []
    visited = set()
    for i in range(matrix.shape[0]):
        if i in visited:
            continue
        cluster = {i}
        for j in range(matrix.shape[0]):
            if i != j and matrix.iloc[i, j] >= threshold:
                cluster.add(j)
        visited |= cluster
        clusters.append({matrix.index[k] for k in cluster})
    return clusters

def compute_symbol_energy(df):
    """Sum of squares per symbol as a proxy for total symbolic energy."""
    return np.linalg.norm(df.to_numpy(), axis=1)

def compute_energy_flow(df):
    """Vector difference between consecutive rows to simulate flow."""
    data = df.to_numpy()
    flow = np.gradient(data, axis=0)
    return flow
