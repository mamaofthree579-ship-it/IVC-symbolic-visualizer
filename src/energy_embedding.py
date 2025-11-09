import numpy as np
import pandas as pd

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize geometric columns for energetic comparison."""
    numeric = df.select_dtypes(include=[np.number])
    normalized = (numeric - numeric.min()) / (numeric.max() - numeric.min() + 1e-9)
    return pd.concat([df[["symbol"]], normalized], axis=1)

def compute_energy_vectors(df: pd.DataFrame) -> np.ndarray:
    """
    Compute symbolic energy vectors from normalized features.
    Energy = weighted combination of curvature, symmetry, and density.
    """
    weights = np.array([0.4, 0.3, 0.3])
    geom = df[["curvature", "symmetry", "density"]].to_numpy()
    return np.dot(geom, weights)

def compute_resonance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute resonance relationships between symbols."""
    data = df[["curvature", "symmetry", "density"]].to_numpy()
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    normalized = data / (norm + 1e-9)
    resonance = np.dot(normalized, normalized.T)
    return pd.DataFrame(resonance, index=df["symbol"], columns=df["symbol"])
