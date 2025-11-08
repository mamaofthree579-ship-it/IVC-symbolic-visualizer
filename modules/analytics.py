import numpy as np

def find_resonant_clusters(matrix, threshold=0.75):
    """
    Identify clusters of symbols with resonance above a certain threshold.
    """
    # Ensure matrix is a NumPy array
    matrix = np.array(matrix)

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
