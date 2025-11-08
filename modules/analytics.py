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

def calculate_symbol_frequencies(symbol_series):
    """
    Calculates occurrence frequency for each symbol in a list or pandas Series.
    Returns a dictionary {symbol: frequency}.
    """
    if isinstance(symbol_series, list):
        symbol_series = pd.Series(symbol_series)
    frequency = symbol_series.value_counts(normalize=True).to_dict()
    return frequency

def resonance_matrix(vector_data, normalize=True):
    """
    Builds a resonance correlation matrix from vector magnitudes.
    Shows how strongly each node vibrates in relation to others.
    """
    magnitudes = np.sqrt(vector_data[:,2]**2 + vector_data[:,3]**2)
    matrix = np.corrcoef(magnitudes)
    if normalize:
        matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    return matrix

def convert_matrix_to_edges(matrix, labels):
    """
    Converts a correlation matrix into symbolic edge list for visualization.
    """
    edges = []
    size = matrix.shape[0]
    for i in range(size):
        for j in range(i+1, size):
            if matrix[i, j] > 0.6:  # significance cutoff
                edges.append((labels[i], labels[j]))
    return edges

def generate_resonance_spectrum(matrix, labels):
    """
    Prepares a labeled spectrum for visualization.
    Returns dict: {label: average resonance value}
    """
    spectrum = {}
    for i, label in enumerate(labels):
        spectrum[label] = float(matrix[i, :].mean())
    return spectrum
