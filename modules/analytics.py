import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def normalize_frequency_matrix(matrix):
    """
    Normalize the input frequency or resonance matrix using MinMax scaling.
    This ensures all values are between 0 and 1.
    """
    matrix = np.array(matrix, dtype=float)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(matrix)
    return normalized


def find_resonant_clusters(matrix, threshold=0.75):
    """
    Identify clusters of symbols with resonance above a certain threshold.
    """
    # Ensure input is a NumPy array
    matrix = np.array(matrix, dtype=float)

    clusters = []
    used = set()

    for i in range(matrix.shape[0]):
        if i in used:
            continue

        cluster = [i]
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > threshold:
                cluster.append(j)
                used.add(j)

        clusters.append(cluster)

    return clusters


def compute_symbol_correlations(df):
    """
    Compute pairwise correlations between symbols (columns in the DataFrame).
    Returns a correlation matrix as a DataFrame.
    """
    corr = df.corr(method="pearson")
    return corr


def reduce_dimensions(matrix, n_components=2):
    """
    Apply PCA to reduce the matrix to 2D (for plotting and visualization).
    Returns a DataFrame with x, y coordinates for each symbol.
    """
    matrix = np.array(matrix, dtype=float)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(matrix)
    return pd.DataFrame(reduced, columns=["x", "y"])


def generate_summary_stats(df):
    """
    Generate basic descriptive statistics for the input dataset.
    """
    stats = df.describe().transpose()
    stats["variance"] = df.var()
    return stats


def find_resonant_paths(matrix, min_strength=0.5):
    """
    Identify all strong resonance links between symbols.
    Returns a list of tuples (symbol_a, symbol_b, strength).
    """
    matrix = np.array(matrix, dtype=float)
    edges = []

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i, j] >= min_strength:
                edges.append((i, j, matrix[i, j]))

    return edges


def detect_frequency_anomalies(matrix, z_thresh=2.5):
    """
    Detect anomalously high or low frequencies using z-scores.
    Returns indices of anomalous entries.
    """
    matrix = np.array(matrix, dtype=float)
    z_scores = (matrix - matrix.mean()) / matrix.std()
    anomalies = np.argwhere(np.abs(z_scores) > z_thresh)
    return anomalies.tolist()
