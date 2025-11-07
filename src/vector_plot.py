import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def draw_vector_field(data, show_colorbar=True, arrow_scale=25, figsize=(6,6)):
    """Draw a quiver plot from data dict returned by vector_data.sample_water_symbol."""
    X = data['X']
    Y = data['Y']
    U = data['U']
    V = data['V']
    mag = data['magnitude']

    fig, ax = plt.subplots(figsize=figsize)
    # scatter base points with low alpha to show field density
    sc = ax.scatter(X, Y, c=mag, cmap=cm.Blues, s=18, alpha=0.6)
    # quiver (vectors) - use normalized vectors for consistent arrow sizes
    # but scale arrow length by magnitude for visual emphasis
    norm = np.maximum(mag, 1e-6)
    U_n = U / norm
    V_n = V / norm
    Q = ax.quiver(X, Y, U_n, V_n, mag, cmap=cm.viridis, scale=arrow_scale, width=0.006, alpha=0.9)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(data['meta']['name'])
    if show_colorbar:
        plt.colorbar(Q, ax=ax, label='vector magnitude')
    plt.tight_layout()
    return fig
