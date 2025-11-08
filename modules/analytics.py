import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def resonance_matrix(data: pd.DataFrame) -> np.ndarray:
    """
    Compute a resonance (similarity) matrix from input data.
    The result is an NxN matrix where each cell indicates 
    how strongly two symbols, words, or entities resonate with each other.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset, where each row is an entity and 
        each column is a feature.

    Returns
    -------
    np.ndarray
        NxN resonance/similarity matrix.
    """
    if data is None or len(data) == 0:
        raise ValueError("Input data is empty or None")

    # Normalize data
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data.select_dtypes(include=[np.number]))

    # Compute cosine similarity as resonance metric
    matrix = cosine_similarity(normalized)
    return matrix


def find_resonant_clusters(matrix: np.ndarray, threshold: float = 0.85):
    """
    Identify clusters of entities that exhibit high resonance.

    Parameters
    ----------
    matrix : np.ndarray
        Resonance matrix (NxN).
    threshold : float
        Minimum resonance value for clustering (default 0.85).

    Returns
    -------
    list[list[int]]
        A list of clusters, each a list of indices.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")

    n = matrix.shape[0]
    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        for j in range(n):
            if i != j and matrix[i, j] >= threshold:
                cluster.append(j)
                visited.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)

    return clusters


def summarize_clusters(data: pd.DataFrame, clusters: list):
    """
    Summarize each cluster with entity names or key stats.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset.
    clusters : list[list[int]]
        Cluster index groups.

    Returns
    -------
    list[dict]
        Summary info for each cluster.
    """
    summaries = []
    for c in clusters:
        subset = data.iloc[c]
        summary = {
            "size": len(c),
            "mean_values": subset.mean().to_dict(),
            "members": subset.index.tolist()
        }
        summaries.append(summary)
    return summaries
