import numpy as np

def spatial_fft(height):
    H = np.fft.fftshift(np.fft.fft2(height))
    return np.log1p(np.abs(H))
