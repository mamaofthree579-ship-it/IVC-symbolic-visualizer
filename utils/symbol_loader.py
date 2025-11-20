import numpy as np

def load_symbol(upload):
    if upload is None:
        return np.ones((512,512)) * 0.5  # fallback synthetic
    return np.load(upload)
