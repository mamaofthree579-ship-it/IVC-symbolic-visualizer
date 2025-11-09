import numpy as np
import pandas as pd

def evolve_energy_step(matrix: pd.DataFrame, damping=0.02, coupling=0.3) -> pd.DataFrame:
    """
    Simulate one step of energy exchange between symbols.
    """
    m = matrix.to_numpy()
    delta = coupling * (m - np.mean(m))
    m_new = m + delta - damping * m
    return pd.DataFrame(m_new, index=matrix.index, columns=matrix.columns)

def compute_energy_density(matrix: pd.DataFrame) -> pd.Series:
    """Compute the average energy of each symbol."""
    return matrix.mean(axis=1)

def detect_energy_stabilization(matrix_sequence):
    """Detect convergence or oscillation in a sequence of matrices."""
    changes = [np.linalg.norm(matrix_sequence[i+1] - matrix_sequence[i])
               for i in range(len(matrix_sequence) - 1)]
    return np.mean(changes[-3:]) < 1e-3  # steady if recent changes small
