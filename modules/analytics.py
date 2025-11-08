# modules/analytics.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def resonance_matrix(data):
    """
    Compute a resonance/similarity matrix from input data.
    Accepts a pandas DataFrame or a 2D numpy array-like of numeric features.
    Returns an NxN numpy.ndarray (cosine similarity).
    """
    # Accept DataFrame or numpy array
    if data is None:
        raise ValueError("No data provided to resonance_matrix()")

    # If DataFrame, select numeric columns and use their values
    if isinstance(data, pd.DataFrame):
        numeric = data.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise ValueError("DataFrame must contain numeric columns")
        arr = numeric.values
    else:
        arr = np.array(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Input array must be 2D (rows=entities, cols=features)")

    # Normalize features to 0-1 range to stabilize cosine similarity
    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(arr)

    # Compute cosine similarity (rows vs rows)
    mat = cosine_similarity(arr_scaled)
    return np.array(mat, dtype=float)


def find_resonant_clusters(matrix, threshold=0.85):
    """
    Identify clusters of indices where pairwise resonance >= threshold.
    Input:
      - matrix: square numpy array (NxN)
      - threshold: float (0..1)
    Returns:
      - list of lists (clusters of indices)
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square 2D array")

    n = matrix.shape[0]
    visited = set()
    clusters = []

    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(n):
            if i == j:
                continue
            if matrix[i, j] >= threshold:
                group.append(j)
                visited.add(j)
        if len(group) > 1:
            clusters.append(sorted(group))
    return clusters


def generate_resonance_spectrum(matrix, labels=None):
    """
    Return average resonance per row as dict {label: value}.
    If labels is None, uses integers as labels.
    """
    mat = np.asarray(matrix, dtype=float)
    avg = mat.mean(axis=1)
    if labels is None:
        labels = [f"Node_{i}" for i in range(mat.shape[0])]
    return {labels[i]: float(avg[i]) for i in range(mat.shape[0])}
