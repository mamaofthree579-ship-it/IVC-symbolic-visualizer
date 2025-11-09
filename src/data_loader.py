import pandas as pd
import numpy as np

def load_symbol_dataset(path: str) -> pd.DataFrame:
    """
    Load a symbol dataset (CSV or TSV).
    Expected columns: 'symbol', 'x', 'y', 'curvature', 'symmetry', 'density'
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep="\t")
    return df

def generate_synthetic_symbols(n=10):
    """
    Generate synthetic symbol data (for testing).
    Each symbol has geometric features that can be mapped to energy.
    """
    np.random.seed(42)
    df = pd.DataFrame({
        "symbol": [f"Symbol_{i}" for i in range(n)],
        "x": np.random.rand(n),
        "y": np.random.rand(n),
        "curvature": np.random.rand(n),
        "symmetry": np.random.rand(n),
        "density": np.random.rand(n)
    })
    return df
