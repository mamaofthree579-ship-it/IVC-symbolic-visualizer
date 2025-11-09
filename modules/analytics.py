import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# Generate Sample Data
# ------------------------------------------------------------------------------
def generate_sample_data(n=5):
    """Generate symbolic test data with pseudo-random energetic correlations."""
    np.random.seed(42)
    symbols = [f"Symbol_{i}" for i in range(n)]
    values = np.random.rand(n, n)
    df = pd.DataFrame(values, columns=symbols, index=symbols)
    return df


# ------------------------------------------------------------------------------
# Resonance Matrix Calculation
# ------------------------------------------------------------------------------
def compute_resonance_matrix(df):
    """
    Compute a resonance matrix showing symbolic interaction strength.
    Uses cosine similarity between symbolic vectors.
    """
    data = df.to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    resonance = np.dot(normalized, normalized.T)
    return pd.DataFrame(resonance, index=df.index, columns=df.columns)


# ------------------------------------------------------------------------------
# Cluster Detection
# ------------------------------------------------------------------------------
def find_resonant_clusters(matrix, threshold=0.8):
    """
    Identify clusters of symbols that resonate above a threshold.
    Returns a list of symbolic sets (clusters).
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


# ------------------------------------------------------------------------------
# Symbolic Energy Mapping
# ------------------------------------------------------------------------------
def compute_symbol_energy(matrix):
    """
    Compute symbolic 'energy' for each symbol.
    Uses sum of resonance strengths as an energy proxy.
    Returns a dictionary for visual mapping.
    """
    energies = matrix.sum(axis=1).to_numpy()
    symbols = matrix.index.tolist()
    energy_map = {symbols[i]: float(energies[i]) for i in range(len(symbols))}
    return energy_map


# ------------------------------------------------------------------------------
# Energy Flow Field (Vector Field)
# ------------------------------------------------------------------------------
def compute_energy_flow(df):
    """
    Compute directional symbolic energy flow between symbols.
    Each vector represents the differential energy interaction.
    """
    data = df.to_numpy()
    grad_x = np.gradient(data, axis=0)
    grad_y = np.gradient(data, axis=1)
    grad_z = grad_x + grad_y  # symbolic “cross-field” flow

    # Flattened field
    flow_vectors = np.mean(
        np.stack([grad_x.flatten(), grad_y.flatten(), grad_z.flatten()], axis=1),
        axis=0
    )
    # Expand to match number of symbols
    n = df.shape[0]
    flow_vectors = np.tile(flow_vectors, (n, 1))
    return flow_vectors


# ------------------------------------------------------------------------------
# Evolving Resonance (for Animation)
# ------------------------------------------------------------------------------
def evolve_matrix_step(matrix, t=0.05):
    """
    Simulate an iterative transformation of the resonance matrix over time.
    Adds oscillatory modulation to reflect dynamic symbol resonance.
    """
    M = matrix.to_numpy()
    noise = np.sin(np.linspace(0, np.pi * 2, M.shape[0]))[:, None]
    evolution = M + t * (np.dot(M, M.T) * noise)
    evolution = np.clip(evolution, 0, 1)
    return pd.DataFrame(evolution, index=matrix.index, columns=matrix.columns)
