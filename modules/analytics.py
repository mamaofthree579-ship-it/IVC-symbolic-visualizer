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
    """
    Compute a resonance matrix showing symbolic interaction strength.
    For now, uses cosine similarity between rows as a proxy for resonance.
    """
    data = df.to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    resonance = np.dot(normalized, normalized.T)
    return pd.DataFrame(resonance, index=df.index, columns=df.columns)

def find_resonant_clusters(matrix, threshold=0.8):
    """
    Find clusters of highly resonant symbols.
    Returns a list of sets, each representing a cluster.
    """
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
